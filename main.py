from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import csv
import gzip
import requests
import base64
from pathlib import Path
from functools import lru_cache
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw

app = FastAPI()

# Allow your published Bolt app and local dev apps to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://interactive-lego-3d-dfsf.bolt.host",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3"


# Local Rebrickable CSV catalog files uploaded to this repo.
# Expected files:
# catalog/parts.csv.gz
# catalog/colors.csv.gz
# catalog/part_categories.csv.gz
# catalog/elements.csv.gz
CATALOG_DIR = Path(__file__).resolve().parent / "catalog"


# ---------------------------------------------------------------------
# Local catalog loading and realistic candidate part filtering
# ---------------------------------------------------------------------

def open_catalog_csv(filename):
    """
    Opens a catalog CSV from /catalog.
    Supports both .csv.gz and .csv filenames.
    """
    gz_path = CATALOG_DIR / f"{filename}.gz"
    csv_path = CATALOG_DIR / filename

    if gz_path.exists():
        return gzip.open(gz_path, "rt", encoding="utf-8", newline="")

    if csv_path.exists():
        return open(csv_path, "r", encoding="utf-8", newline="")

    return None


@lru_cache(maxsize=1)
def load_local_rebrickable_catalog():
    """
    Loads Rebrickable CSV downloads from the local /catalog folder.

    This local catalog is used for fast product-safe filtering.
    Rebrickable API is still used separately for validation/enrichment when requested.
    """
    parts_by_num = {}
    categories_by_id = {}
    colors_by_id = {}
    elements_by_part_color = set()

    # part_categories.csv: id,name
    handle = open_catalog_csv("part_categories.csv")
    if handle:
        with handle:
            for row in csv.DictReader(handle):
                cat_id = str(row.get("id", "")).strip()
                if not cat_id:
                    continue
                categories_by_id[cat_id] = {
                    "id": cat_id,
                    "name": row.get("name", "")
                }

    # parts.csv: part_num,name,part_cat_id,part_material
    handle = open_catalog_csv("parts.csv")
    if handle:
        with handle:
            for row in csv.DictReader(handle):
                part_num = str(row.get("part_num", "")).strip()
                if not part_num:
                    continue

                cat_id = str(row.get("part_cat_id", "")).strip()
                category = categories_by_id.get(cat_id, {})

                parts_by_num[part_num] = {
                    "part_num": part_num,
                    "name": row.get("name", ""),
                    "part_cat_id": cat_id,
                    "part_category": category.get("name"),
                    "part_material": row.get("part_material", "")
                }

    # colors.csv: id,name,rgb,is_trans
    handle = open_catalog_csv("colors.csv")
    if handle:
        with handle:
            for row in csv.DictReader(handle):
                color_id = str(row.get("id", "")).strip()
                if not color_id:
                    continue
                colors_by_id[color_id] = {
                    "id": color_id,
                    "name": row.get("name", ""),
                    "rgb": row.get("rgb", ""),
                    "is_trans": row.get("is_trans", "")
                }

    # elements.csv: element_id,part_num,color_id
    # This tells us that a part/color combination exists as an official LEGO element.
    handle = open_catalog_csv("elements.csv")
    if handle:
        with handle:
            for row in csv.DictReader(handle):
                part_num = str(row.get("part_num", "")).strip()
                color_id = str(row.get("color_id", "")).strip()
                if part_num and color_id:
                    elements_by_part_color.add((part_num, color_id))

    return {
        "catalog_dir": str(CATALOG_DIR),
        "catalog_available": bool(parts_by_num),
        "parts_by_num": parts_by_num,
        "categories_by_id": categories_by_id,
        "colors_by_id": colors_by_id,
        "elements_by_part_color": elements_by_part_color,
        "counts": {
            "parts": len(parts_by_num),
            "categories": len(categories_by_id),
            "colors": len(colors_by_id),
            "elements_part_color_pairs": len(elements_by_part_color)
        }
    }


def get_catalog_health():
    catalog = load_local_rebrickable_catalog()
    return {
        "catalog_dir": catalog["catalog_dir"],
        "catalog_available": catalog["catalog_available"],
        "counts": catalog["counts"],
        "expected_files": [
            "catalog/parts.csv.gz",
            "catalog/colors.csv.gz",
            "catalog/part_categories.csv.gz",
            "catalog/elements.csv.gz"
        ]
    }


def parse_stud_size_from_name(part_name):
    """
    Extracts simple rectangular stud size from Rebrickable-style names.
    Examples:
    - Plate 1 x 2 -> w=1, h=2
    - Tile 2 x 4 with Groove -> w=2, h=4
    """
    name = str(part_name or "")
    match = re.search(r"(\d+)\s*x\s*(\d+)", name, flags=re.IGNORECASE)

    if not match:
        return None, None

    try:
        return int(match.group(1)), int(match.group(2))
    except Exception:
        return None, None


def is_unrealistic_product_part(part_num, part_name, category_name):
    """
    Filters out parts that are real LEGO parts but poor choices for an automated product kit.
    """
    num = str(part_num or "").lower()
    name = normalize_text(part_name)
    category = normalize_text(category_name)

    # Printed/decorated variants often include pr/pb/sticker-specific variants.
    # For automated kits, prefer plain parts unless manually approved.
    if "pr" in num or "sticker" in name or "pattern" in name or "printed" in name:
        return True

    blocked_category_terms = [
        "minifig", "animal", "duplo", "electric", "energy", "gear",
        "wheel", "tyre", "tire", "technic", "pneumatic", "string",
        "sticker", "weapon", "tool", "container", "belville", "bionicle",
        "hero factory", "znap", "modulex", "baseplate", "large buildable figure"
    ]

    if any(term in category for term in blocked_category_terms):
        return True

    blocked_name_terms = [
        "minifig", "duplo", "sticker", "electric", "wheel", "tyre", "tire",
        "weapon", "helmet", "head", "torso", "leg", "arm", "cape", "cloth",
        "string", "rubber band", "hose", "net", "animal", "food", "book", "flag"
    ]

    if any(term in name for term in blocked_name_terms):
        return True

    return False


def make_candidate_record(part_name, part_num, role, source="strategy_allowed_parts", preferred=True):
    catalog = load_local_rebrickable_catalog()
    local_part = catalog["parts_by_num"].get(str(part_num), {})

    official_name = local_part.get("name") or part_name
    category_name = local_part.get("part_category")
    w, h = parse_stud_size_from_name(official_name)

    return {
        "part_name": part_name,
        "part_num": str(part_num),
        "official_part_name": official_name,
        "part_category": category_name,
        "w": w,
        "h": h,
        "role": role,
        "preferred": preferred,
        "source": source,
        "local_catalog_match": bool(local_part),
        "product_safe": not is_unrealistic_product_part(part_num, official_name, category_name)
    }


def classify_allowed_part_role(part_name, part_num, strategy):
    name = normalize_text(part_name)

    if "tile round" in name or "plate round" in name or "round" in name:
        return "detail"

    if "tile" in name:
        return "surface"

    if "plate" in name:
        return "structure"

    if "slope" in name or "curved" in name or "arch" in name or "masonry" in name or "headlight" in name:
        return "shaping"

    if "plant" in name:
        return "detail"

    if "brick" in name:
        return "structure"

    return "other"


def discover_extra_catalog_candidates(strategy, limit_per_role=25):
    """
    Finds extra realistic product-safe parts from the local Rebrickable CSV catalog.
    This is not a blind all-parts dump. It only includes category/name patterns
    that are useful for the selected build strategy.
    """
    catalog = load_local_rebrickable_catalog()
    parts_by_num = catalog["parts_by_num"]

    discovered = {
        "surface": [],
        "detail": [],
        "structure": [],
        "shaping": []
    }

    strategy = str(strategy or "object_depth_to_voxel")

    for part_num, part in parts_by_num.items():
        official_name = part.get("name", "")
        category_name = part.get("part_category", "")
        norm_name = normalize_text(official_name)
        norm_cat = normalize_text(category_name)

        if is_unrealistic_product_part(part_num, official_name, category_name):
            continue

        w, h = parse_stud_size_from_name(official_name)

        # Avoid huge parts for automated image-to-model products.
        if w and h and (w > 8 or h > 8):
            continue

        role = None

        if strategy in ["mosaic_or_relief_conversion", "portrait_bust_template", "pet_template_customization"]:
            if "tiles" in norm_cat and ("tile" in norm_name):
                role = "surface"
            elif "plates round" in norm_cat or "round" in norm_name:
                if "plate" in norm_name or "tile" in norm_name:
                    role = "detail"
            elif "plates" in norm_cat and "plate" in norm_name:
                role = "structure"
            elif "slope" in norm_name or "curved" in norm_name:
                role = "shaping"

        elif strategy == "architecture_studio_rebuild":
            if "brick" in norm_name or "masonry" in norm_name or "arch" in norm_name or "panel" in norm_name:
                role = "structure"
            elif "plate" in norm_name:
                role = "structure"
            elif "tile" in norm_name:
                role = "surface"
            elif "slope" in norm_name or "curved" in norm_name:
                role = "shaping"

        elif strategy == "vehicle_template_customization":
            if "plate" in norm_name:
                role = "structure"
            elif "tile" in norm_name:
                role = "surface"
            elif "slope" in norm_name or "curved" in norm_name:
                role = "shaping"

        elif strategy == "landscape_diorama_rebuild":
            if "plate" in norm_name:
                role = "structure"
            elif "tile" in norm_name:
                role = "surface"
            elif "plant" in norm_name or "flower" in norm_name or "leaf" in norm_name:
                role = "detail"
            elif "slope" in norm_name or "curved" in norm_name:
                role = "shaping"

        else:
            if "plate" in norm_name or "brick" in norm_name:
                role = "structure"
            elif "tile" in norm_name:
                role = "surface"
            elif "round" in norm_name:
                role = "detail"
            elif "slope" in norm_name or "curved" in norm_name:
                role = "shaping"

        if not role or role not in discovered:
            continue

        if len(discovered[role]) >= limit_per_role:
            continue

        discovered[role].append({
            "part_name": official_name,
            "part_num": part_num,
            "official_part_name": official_name,
            "part_category": category_name,
            "w": w,
            "h": h,
            "role": role,
            "preferred": False,
            "source": "local_rebrickable_csv_discovery",
            "local_catalog_match": True,
            "product_safe": True
        })

    return discovered


def build_candidate_parts_catalog(analysis, data=None):
    """
    Builds a realistic candidate catalog for Server 2.

    Important:
    - It does not send all Rebrickable parts.
    - It uses strategy-safe parts as the optimizer-ready catalog.
    - It checks/enriches those parts against the local Rebrickable CSV catalog.
    - Extra discovered CSV parts are kept only as reference unless explicitly requested.
    """
    data = data or {}
    strategy = get_build_strategy(analysis)
    allowed_parts = get_allowed_parts_for_analysis(analysis)
    catalog_health = get_catalog_health()

    surface_parts = []
    detail_parts = []
    structure_parts = []
    shaping_parts = []
    other_parts = []

    optimizer_ready_parts = []
    seen = set()

    for part_name, part_num in allowed_parts.items():
        role = classify_allowed_part_role(part_name, part_num, strategy)
        record = make_candidate_record(
            part_name=part_name,
            part_num=part_num,
            role=role,
            source="strategy_allowed_parts",
            preferred=True
        )

        # Only strategy-allowed, locally matched, product-safe records are optimizer-ready.
        record["optimizer_ready"] = bool(
            record.get("preferred")
            and record.get("local_catalog_match")
            and record.get("product_safe")
            and record.get("w") is not None
            and record.get("h") is not None
        )

        seen.add(str(part_num))

        if role == "surface":
            surface_parts.append(record)
        elif role == "detail":
            detail_parts.append(record)
        elif role == "structure":
            structure_parts.append(record)
        elif role == "shaping":
            shaping_parts.append(record)
        else:
            other_parts.append(record)

        if record["optimizer_ready"]:
            optimizer_ready_parts.append(record)

    # Discovery is disabled by default for production safety.
    # Extra discovered parts can be returned for reference/debugging only, but Server 2 should not use them.
    include_discovered = bool(data.get("include_catalog_discovery", False))
    discovered_reference_parts = []

    if include_discovered and catalog_health["catalog_available"]:
        discovered = discover_extra_catalog_candidates(strategy)

        for role, records in discovered.items():
            for record in records:
                part_num = str(record.get("part_num"))
                if part_num in seen:
                    continue
                seen.add(part_num)

                record["optimizer_ready"] = False
                record["server_2_use"] = "reference_only_not_for_optimizer"
                discovered_reference_parts.append(record)

    # These grouped lists intentionally contain only optimizer-ready strategy parts.
    safe_surface_parts = [p for p in surface_parts if p.get("optimizer_ready")]
    safe_detail_parts = [p for p in detail_parts if p.get("optimizer_ready")]
    safe_structure_parts = [p for p in structure_parts if p.get("optimizer_ready")]
    safe_shaping_parts = [p for p in shaping_parts if p.get("optimizer_ready")]
    safe_other_parts = [p for p in other_parts if p.get("optimizer_ready")]

    return {
        "strategy": strategy,
        "catalog_source": "strategy_allowed_parts_plus_local_rebrickable_csv_validation",
        "local_catalog_health": catalog_health,
        "selection_policy": {
            "use_all_rebrickable_parts_blindly": False,
            "filter_by_strategy": True,
            "product_safe_filtering": True,
            "optimizer_uses_strategy_allowed_parts_only": True,
            "local_csv_discovery_enabled": include_discovered,
            "local_csv_discovery_usage": "reference_only_not_optimizer_ready",
            "preferred_surface_style": data.get("preferred_surface_style", "smooth_tiles"),
            "allow_printed_parts": False,
            "allow_minifig_parts": False,
            "allow_vehicle_only_parts_for_mosaic": False,
            "notes": [
                "The full Rebrickable CSV catalog is used for lookup and validation, not blindly sent to the optimizer.",
                "Server 2 should use optimizer_ready_parts only.",
                "Discovered CSV parts are reference-only unless manually approved later."
            ]
        },
        "surface_parts": safe_surface_parts,
        "detail_parts": safe_detail_parts,
        "structure_parts": safe_structure_parts,
        "shaping_parts": safe_shaping_parts,
        "other_parts": safe_other_parts,
        "optimizer_ready_parts": optimizer_ready_parts,
        "optimizer_ready_part_count": len(optimizer_ready_parts),
        "discovered_reference_parts": discovered_reference_parts,
        "discovered_reference_part_count": len(discovered_reference_parts),
        "server_2_usage_note": "Send candidate_parts_catalog.optimizer_ready_parts with image_geometry to the GLB/model server. Server 2 should ignore discovered_reference_parts unless manually approved."
    }


@app.get("/catalog/health")
def catalog_health():
    return get_catalog_health()


# Strategy-specific allowed parts.
# These are controlled part libraries, not blind Rebrickable searches.
# Rebrickable is used to validate these exact part numbers and colors.
#
# Part IDs are taken from your RebrickNet extraction and common Rebrickable part IDs.
# Do not add every scraped part to every strategy. Keep parts grouped by build use.
ALLOWED_PARTS_BY_STRATEGY = {
    "mosaic_or_relief_conversion": {
        # Core mosaic tiles
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 1x3": "63864",
        "Tile 1x4": "2431",
        "Tile 1x6": "6636",
        "Tile 1x8": "4162",
        "Tile 2x2": "3068b",
        "Tile 2x3": "26603",
        "Tile 2x4": "87079",

        # Round detail tiles
        "Tile Round 1x1": "98138",
        "Tile Round 1x1 Half Circle": "24246",
        "Tile Round 1x1 Quarter": "25269",
        "Tile Round 2x2": "14769",

        # Plates for relief/depth
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 1x3": "3623",
        "Plate 1x4": "3710",
        "Plate 2x2": "3022",
        "Plate 2x3": "3021",
        "Plate 2x4": "3020",

        # Small round highlights
        "Plate Round 1x1 Open Stud": "85861",
        "Plate Round 1x1 Solid Stud": "6141"
    },

    "pet_template_customization": {
        # Mosaic/relief surface
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 1x3": "63864",
        "Tile 1x4": "2431",
        "Tile 2x2": "3068b",
        "Tile Round 1x1": "98138",
        "Tile Round 1x1 Half Circle": "24246",
        "Tile Round 1x1 Quarter": "25269",

        # Plates
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 1x3": "3623",
        "Plate 1x4": "3710",
        "Plate 2x2": "3022",
        "Plate 2x3": "3021",
        "Plate 2x4": "3020",
        "Plate Round 1x1 Open Stud": "85861",
        "Plate Round 1x1 Solid Stud": "6141",

        # Small 3D shaping
        "Brick 1x1": "3005",
        "Brick 1x2": "3004",
        "Brick Round 1x1": "3062b",
        "Slope Cheese 1x1": "54200",
        "Slope Curved 2x1": "11477"
    },

    "portrait_bust_template": {
        # Face/skin/facial detail surface
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 1x3": "63864",
        "Tile 1x4": "2431",
        "Tile 2x2": "3068b",
        "Tile Round 1x1": "98138",
        "Tile Round 1x1 Half Circle": "24246",
        "Tile Round 1x1 Quarter": "25269",

        # Relief plates
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 1x3": "3623",
        "Plate 1x4": "3710",
        "Plate 2x2": "3022",
        "Plate 2x3": "3021",
        "Plate 2x4": "3020",
        "Plate Round 1x1 Open Stud": "85861",
        "Plate Round 1x1 Solid Stud": "6141",

        # Shaping parts
        "Brick 1x1": "3005",
        "Brick 1x2": "3004",
        "Slope Cheese 1x1": "54200",
        "Slope Curved 2x1": "11477",
        "Slope Curved 2x2": "15068"
    },

    "architecture_studio_rebuild": {
        # Basic bricks
        "Brick 1x1": "3005",
        "Brick 1x2": "3004",
        "Brick 1x3": "3622",
        "Brick 1x4": "3010",
        "Brick 1x6": "3009",
        "Brick 1x8": "3008",
        "Brick 2x2": "3003",
        "Brick 2x3": "3002",
        "Brick 2x4": "3001",
        "Brick 2x6": "2456",
        "Brick 2x8": "3007",

        # Plates
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 1x3": "3623",
        "Plate 1x4": "3710",
        "Plate 1x6": "3666",
        "Plate 1x8": "3460",
        "Plate 2x2": "3022",
        "Plate 2x3": "3021",
        "Plate 2x4": "3020",
        "Plate 2x6": "3795",
        "Plate 2x8": "3034",
        "Plate 4x4": "3031",
        "Plate 4x6": "3032",
        "Plate 4x8": "3035",

        # Tiles
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 1x4": "2431",
        "Tile 2x2": "3068b",

        # Architecture details
        "Brick Round 1x1": "3062b",
        "Brick Round 2x2": "3941",
        "Brick Round Corner 2x2": "85080",
        "Arch 1x3x2": "6005",
        "Arch 1x4": "3659",
        "Curved Brick 1x2": "6091",
        "Masonry Brick 1x2": "98283",
        "Headlight Brick": "4070a",
        "Brick Studs on 1 Side": "87087",
        "Brick Studs on 2 Sides": "47905",
        "Panel 1x2x1": "4865b",
        "Panel 1x4x1": "15207",

        # Slopes/roof/detail shaping
        "Slope Cheese 1x1": "54200",
        "Slope 1x2": "3040b",
        "Slope 2x2": "3039",
        "Slope 2x3": "3038",
        "Slope 2x4": "3037",
        "Slope Curved 2x1": "11477",
        "Slope Curved 2x2": "15068",
        "Slope Curved 4x1": "93273"
    },

    "vehicle_template_customization": {
        # Body bricks
        "Brick 1x1": "3005",
        "Brick 1x2": "3004",
        "Brick 1x3": "3622",
        "Brick 1x4": "3010",
        "Brick 1x6": "3009",
        "Brick 2x2": "3003",
        "Brick 2x3": "3002",
        "Brick 2x4": "3001",

        # Chassis plates
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 1x3": "3623",
        "Plate 1x4": "3710",
        "Plate 2x2": "3022",
        "Plate 2x3": "3021",
        "Plate 2x4": "3020",
        "Plate 2x6": "3795",
        "Plate 2x8": "3034",
        "Plate 4x4": "3031",
        "Plate 4x6": "3032",

        # Smooth body details
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 1x4": "2431",
        "Tile 2x2": "3068b",
        "Tile 2x4": "87079",

        # Slopes and curves
        "Slope Cheese 1x1": "54200",
        "Slope 1x2": "3040b",
        "Slope 2x2": "3039",
        "Slope Curved 2x1": "11477",
        "Slope Curved 2x2": "15068",
        "Slope Curved 3x1": "50950",
        "Slope Curved 4x1": "93273",

        # Vehicle-specific functional pieces
        "Plate Wheel Holder 2x2": "67687",
        "Steering Wheel Assembly": "3829c01",
        "Bar 3L": "87994",
        "Bar 4L": "30374",
        "Technic Axle 4": "3705",
        "Technic Axle 6": "3706",
        "Technic Pin": "3673"
    },

    "object_depth_to_voxel": {
        # General bricks
        "Brick 1x1": "3005",
        "Brick 1x2": "3004",
        "Brick 1x3": "3622",
        "Brick 1x4": "3010",
        "Brick 1x6": "3009",
        "Brick 2x2": "3003",
        "Brick 2x3": "3002",
        "Brick 2x4": "3001",

        # Plates
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 1x3": "3623",
        "Plate 1x4": "3710",
        "Plate 2x2": "3022",
        "Plate 2x3": "3021",
        "Plate 2x4": "3020",
        "Plate 2x6": "3795",

        # Surface tiles
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 1x4": "2431",
        "Tile 2x2": "3068b",
        "Tile Round 1x1": "98138",

        # Shape helpers
        "Brick Round 1x1": "3062b",
        "Brick Round 2x2": "3941",
        "Slope Cheese 1x1": "54200",
        "Slope 1x2": "3040b",
        "Slope 2x2": "3039",
        "Slope Curved 2x1": "11477",
        "Slope Curved 2x2": "15068"
    },

    "landscape_diorama_rebuild": {
        # Ground plates
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 1x3": "3623",
        "Plate 1x4": "3710",
        "Plate 2x2": "3022",
        "Plate 2x3": "3021",
        "Plate 2x4": "3020",
        "Plate 2x6": "3795",
        "Plate 4x4": "3031",
        "Plate 4x6": "3032",
        "Plate 4x8": "3035",

        # Surface tiles
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 1x4": "2431",
        "Tile 2x2": "3068b",
        "Tile Round 1x1": "98138",

        # Terrain shaping
        "Brick 1x1": "3005",
        "Brick 1x2": "3004",
        "Brick 1x4": "3010",
        "Slope Cheese 1x1": "54200",
        "Slope 1x2": "3040b",
        "Slope 2x2": "3039",
        "Slope Curved 2x1": "11477",
        "Slope Curved 2x2": "15068",

        # Plants / nature details
        "Plant Flower 4 Petals": "33291",
        "Plant Flower 5 Petals": "24866",
        "Plant Flower Stem": "3741",
        "Plant Leaves 4x3": "2423",
        "Plant 1x1 Round 3 Leaves": "32607"
    },

    "manual_review_required": {
        # Safe fallback parts only
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 2x2": "3068b",
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 2x2": "3022",
        "Plate Round 1x1 Solid Stud": "6141",
        "Tile Round 1x1": "98138"
    }
}


@app.get("/")
def root():
    return {
        "status": "running",
        "mode": "analysis_plus_optional_image_geometry",
        "catalog_source": "rebrickable_csv_plus_api_validation",
        "local_catalog": get_catalog_health(),
        "rebrickable_api_configured": rebrickable_is_configured(),
        "external_catalog_api_used": "rebrickable",
        "note": "Colors are resolved through Rebrickable API. Candidate parts are filtered from local Rebrickable CSV files by build strategy, then validated/enriched with Rebrickable when requested. If original_image_url is sent with include_image_geometry=true, the server creates image-based placement geometry. If include_preview_image=true, the server returns a 2D stud preview as base64 PNG."
    }


def rebrickable_is_configured():
    return bool(os.environ.get("REBRICKABLE_API_KEY"))


def rebrickable_headers():
    api_key = os.environ.get("REBRICKABLE_API_KEY")

    if not api_key:
        raise RuntimeError("Missing REBRICKABLE_API_KEY environment variable.")

    return {
        "Authorization": f"key {api_key}"
    }


def rebrickable_get(path, params=None):
    url = f"{REBRICKABLE_BASE_URL}{path}"

    response = requests.get(
        url,
        headers=rebrickable_headers(),
        params=params,
        timeout=20
    )

    response.raise_for_status()
    return response.json()


@app.get("/rebrickable/test")
def test_rebrickable():
    if not rebrickable_is_configured():
        return {
            "ok": False,
            "error": "Missing REBRICKABLE_API_KEY in Render environment variables."
        }

    try:
        data = rebrickable_get("/lego/colors/", params={"page_size": 5})

        return {
            "ok": True,
            "message": "Rebrickable API connected successfully.",
            "sample_color_count": len(data.get("results", [])),
            "sample_colors": data.get("results", [])
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


@app.get("/rebrickable/colors")
def get_rebrickable_colors():
    if not rebrickable_is_configured():
        return {
            "ok": False,
            "error": "Missing REBRICKABLE_API_KEY in Render environment variables."
        }

    try:
        color_cache = get_rebrickable_colors_cache()

        return {
            "ok": True,
            "count": len(color_cache["colors"]),
            "colors": color_cache["colors"]
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


def normalize_text(value):
    value = str(value or "").lower().strip()
    value = value.replace("-", " ")
    value = value.replace("_", " ")
    value = re.sub(r"[^a-z0-9 ]+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_color_alias(color):
    color = normalize_text(color)

    aliases = {
        "light brown": "tan",
        "beige": "tan",
        "cream": "tan",
        "sand": "tan",

        "dark brown": "reddish brown",
        "reddish brown": "reddish brown",

        "grey": "light bluish gray",
        "gray": "light bluish gray",
        "light grey": "light bluish gray",
        "light gray": "light bluish gray",
        "light bluish grey": "light bluish gray",

        "dark grey": "dark bluish gray",
        "dark gray": "dark bluish gray",
        "dark bluish grey": "dark bluish gray",

        "pale pink": "red",
        "pink": "red",

        "light green": "lime",
        "bright green": "lime",
        "bright light green": "bright light green",
        "hazel": "lime",
        "green eyes": "lime",
        "light green eyes": "lime",
        "hazel eyes": "lime",

        "dark green": "dark green",

        # Extra architecture / stone aliases
        "very light gray": "very light gray",
        "very light grey": "very light gray",
        "medium gray": "light bluish gray",
        "medium grey": "light bluish gray",
        "dark stone gray": "dark bluish gray",
        "dark stone grey": "dark bluish gray",
        "medium stone gray": "light bluish gray",
        "medium stone grey": "light bluish gray",

        "light tan": "tan",
        "dark tan": "dark tan",
        "sand yellow": "tan",
        "sand beige": "tan",
        "masonry tan": "tan",
        "nougat": "nougat",
        "medium nougat": "medium nougat",

        "brown": "brown",
        "dark brown": "dark brown",

        "sand green": "sand green",
        "sand blue": "sand blue",
        "light blue": "medium blue",
        "sky blue": "medium blue",
        "dark red": "dark red"
    }

    return aliases.get(color, color)


@lru_cache(maxsize=1)
def get_rebrickable_colors_cache():
    """
    Loads all Rebrickable colors once and keeps them in memory.
    This avoids hardcoding color IDs.
    """
    colors = []
    page = 1

    while True:
        data = rebrickable_get(
            "/lego/colors/",
            params={
                "page": page,
                "page_size": 1000
            }
        )

        colors.extend(data.get("results", []))

        if not data.get("next"):
            break

        page += 1

    by_normalized_name = {}

    for color in colors:
        color_id = color.get("id")
        name = color.get("name")

        if color_id is None or not name:
            continue

        by_normalized_name[normalize_text(name)] = {
            "id": color_id,
            "name": name,
            "rgb": color.get("rgb"),
            "is_trans": color.get("is_trans"),
            "num_parts": color.get("num_parts"),
            "num_sets": color.get("num_sets"),
            "first_year": color.get("first_year"),
            "last_year": color.get("last_year"),
            "external_ids": color.get("external_ids")
        }

    return {
        "colors": colors,
        "by_normalized_name": by_normalized_name
    }


def resolve_rebrickable_color(color_name):
    """
    Converts planner color names like 'tan', 'lime', or 'light_bluish_gray'
    into actual Rebrickable color records from the API.
    """
    normalized = normalize_color_alias(color_name)
    color_map = get_rebrickable_colors_cache()["by_normalized_name"]

    if normalized in color_map:
        return color_map[normalized]

    fallback = color_map.get("light bluish gray")

    if fallback:
        return fallback

    return {
        "id": None,
        "name": color_name,
        "rgb": None,
        "is_trans": False,
        "num_parts": None,
        "num_sets": None,
        "first_year": None,
        "last_year": None,
        "external_ids": None
    }


def analysis_text_blob(analysis):
    parts = []

    keys = [
        "subject",
        "category",
        "scene_type",
        "viewing_angle",
        "camera_crop",
        "pose_or_orientation",
        "approximate_complexity",
        "recommended_model_type",
        "build_strategy",
        "automation_level",
        "product_title",
        "short_product_description"
    ]

    for key in keys:
        value = analysis.get(key)
        if isinstance(value, str):
            parts.append(value)

    list_keys = [
        "main_objects",
        "secondary_objects",
        "background_elements",
        "foreground_elements",
        "visible_materials",
        "dominant_shapes",
        "color_palette",
        "texture_details",
        "important_details_to_preserve",
        "relative_object_positions",
        "brick_conversion_notes",
        "build_challenges"
    ]

    for key in list_keys:
        value = analysis.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)

    structure_geometry = analysis.get("structure_geometry", {})
    if isinstance(structure_geometry, dict):
        for value in structure_geometry.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value)

    return normalize_text(" ".join(parts))


def choose_eye_color(analysis):
    """
    Chooses a better eye color for pet/portrait mosaic details.
    This uses the image analysis text, then resolves the color dynamically
    through Rebrickable colors API.
    """
    text = analysis_text_blob(analysis)

    if "light green" in text or "bright green" in text or "green eye" in text or "green eyes" in text:
        return "lime"

    if "hazel" in text:
        return "lime"

    if "yellow eye" in text or "yellow eyes" in text or "amber eye" in text or "amber eyes" in text:
        return "yellow"

    if "blue eye" in text or "blue eyes" in text:
        return "medium blue"

    if "brown eye" in text or "brown eyes" in text:
        return "reddish_brown"

    return "green"


def choose_primary_color(analysis, fallback="light_bluish_gray"):
    text = analysis_text_blob(analysis)

    if "black" in text:
        return "black"
    if "white" in text:
        return "white"
    if "red" in text:
        return "red"
    if "blue" in text:
        return "blue"
    if "green" in text:
        return "green"
    if "yellow" in text:
        return "yellow"
    if "orange" in text:
        return "orange"
    if "tan" in text or "beige" in text or "cream" in text:
        return "tan"
    if "dark gray" in text or "dark grey" in text or "dark stone" in text:
        return "dark_bluish_gray"
    if "gray" in text or "grey" in text:
        return "light_bluish_gray"
    if "brown" in text:
        return "reddish_brown"

    return fallback


def get_build_strategy(analysis):
    strategy = str(analysis.get("build_strategy", "")).strip()

    if strategy:
        return strategy

    model_type = str(analysis.get("recommended_model_type", "")).strip()
    category = str(analysis.get("category", "")).strip()

    if model_type == "mosaic_relief":
        return "mosaic_or_relief_conversion"

    if model_type in ["architecture_full_model", "architecture_facade"]:
        return "architecture_studio_rebuild"

    if model_type == "vehicle_model":
        return "vehicle_template_customization"

    if category in ["pet_animal"]:
        return "pet_template_customization"

    if category in ["person_portrait"]:
        return "portrait_bust_template"

    if category in ["landscape_scene"]:
        return "landscape_diorama_rebuild"

    if category in ["vehicle"]:
        return "vehicle_template_customization"

    return "object_depth_to_voxel"


def get_allowed_parts_for_analysis(analysis):
    strategy = get_build_strategy(analysis)

    return ALLOWED_PARTS_BY_STRATEGY.get(
        strategy,
        ALLOWED_PARTS_BY_STRATEGY["object_depth_to_voxel"]
    )


def resolve_part(part_name, analysis):
    allowed_parts = get_allowed_parts_for_analysis(analysis)
    part_num = allowed_parts.get(part_name)

    if not part_num:
        return {
            "part_num": None,
            "official_part_name": None,
            "part_img_url": None,
            "error": f"Part '{part_name}' is not allowed for strategy '{get_build_strategy(analysis)}'."
        }

    return {
        "part_num": part_num,
        "official_part_name": None,
        "part_img_url": None,
        "error": None
    }


def get_tier_multiplier(analysis):
    tier = str(analysis.get("recommended_pricing_tier", "")).lower()

    if tier == "premium":
        return 3

    if tier == "standard":
        return 2

    return 1


def add_part_line(parts, part_name, color, qty, analysis, module_id=None, purpose=None):
    color_data = resolve_rebrickable_color(color)
    part_data = resolve_part(part_name, analysis)

    part_num = part_data.get("part_num")
    resolve_error = part_data.get("error")

    notes = []

    if resolve_error:
        notes.append(resolve_error)

    if not part_num:
        notes.append(f"Could not resolve allowed part for planner part: {part_name}")

    if color_data.get("id") is None:
        notes.append(f"Could not resolve color from Rebrickable colors API for planner color: {color}")

    parts.append({
        "part_name": part_name,
        "part_num": part_num,
        "color": color,
        "rebrickable_color_id": color_data.get("id"),
        "rebrickable_color_name": color_data.get("name"),
        "rebrickable_color_rgb": color_data.get("rgb"),
        "rebrickable_color_external_ids": color_data.get("external_ids"),
        "quantity": int(qty),
        "strategy": get_build_strategy(analysis),
        "module_id": module_id,
        "purpose": purpose,
        "validation": {
            "checked": False,
            "part_exists": None,
            "color_exists_for_part": None,
            "official_part_name": None,
            "notes": notes
        }
    })


def build_parts_summary(parts):
    summary = {}

    for p in parts:
        color_name = p.get("rebrickable_color_name") or p.get("color")
        key = f"{p['part_name']} - {color_name}"
        summary[key] = summary.get(key, 0) + int(p["quantity"])

    return summary


def create_rebrickable_parts_export(parts):
    rows = []

    for p in parts:
        if not p.get("part_num") or p.get("rebrickable_color_id") is None:
            continue

        rows.append({
            "part_num": p["part_num"],
            "color_id": p["rebrickable_color_id"],
            "color_name": p.get("rebrickable_color_name"),
            "quantity": p["quantity"]
        })

    return rows


def create_basic_xml_export(parts):
    """
    Generic XML export using Rebrickable-resolved part numbers and Rebrickable color IDs.
    This is not BrickLink XML.
    """
    xml = "<INVENTORY>\n"

    for p in parts:
        part_num = p.get("part_num")
        color_id = p.get("rebrickable_color_id")

        if not part_num or color_id is None:
            continue

        xml += "  <ITEM>\n"
        xml += f"    <PARTNUM>{part_num}</PARTNUM>\n"
        xml += f"    <COLORID>{color_id}</COLORID>\n"
        xml += f"    <COLORNAME>{p.get('rebrickable_color_name')}</COLORNAME>\n"
        xml += f"    <QTY>{p['quantity']}</QTY>\n"
        xml += "  </ITEM>\n"

    xml += "</INVENTORY>"
    return xml


def validate_parts_with_rebrickable(parts):
    if not rebrickable_is_configured():
        for p in parts:
            p["validation"]["checked"] = False
            p["validation"]["notes"].append("Rebrickable API key is not configured.")
        return parts

    for p in parts:
        part_num = p.get("part_num")
        color_id = p.get("rebrickable_color_id")

        if not part_num:
            p["validation"]["checked"] = True
            p["validation"]["part_exists"] = False
            p["validation"]["color_exists_for_part"] = False
            p["validation"]["notes"].append("Missing Rebrickable part number.")
            continue

        try:
            part_data = rebrickable_get(f"/lego/parts/{part_num}/")

            p["validation"]["checked"] = True
            p["validation"]["part_exists"] = True
            p["validation"]["official_part_name"] = part_data.get("name")

            if part_data.get("part_img_url"):
                p["part_img_url"] = part_data.get("part_img_url")

        except Exception as e:
            p["validation"]["checked"] = True
            p["validation"]["part_exists"] = False
            p["validation"]["color_exists_for_part"] = False
            p["validation"]["notes"].append(f"Part lookup failed: {str(e)}")
            continue

        if color_id is None:
            p["validation"]["color_exists_for_part"] = False
            p["validation"]["notes"].append("Missing Rebrickable color ID.")
            continue

        try:
            color_data = rebrickable_get(f"/lego/parts/{part_num}/colors/{color_id}/")

            p["validation"]["color_exists_for_part"] = True
            p["validation"]["notes"].append("Part/color combination found in Rebrickable.")

            if color_data.get("part_img_url"):
                p["part_img_url"] = color_data.get("part_img_url")

            if color_data.get("elements"):
                p["elements"] = color_data.get("elements")

        except Exception as e:
            p["validation"]["color_exists_for_part"] = False
            p["validation"]["notes"].append(
                f"Part/color combination not found or not common: {str(e)}"
            )

    return parts


def make_response(message, analysis, build_modules, parts, data):
    validate = bool(data.get("validate_with_rebrickable", True))
    include_export = bool(data.get("include_basic_xml_export", True))

    if validate:
        parts = validate_parts_with_rebrickable(parts)

    response = {
        "message": message,
        "generation_mode": "analysis_only",
        "catalog_source": "rebrickable",
        "rebrickable_api_configured": rebrickable_is_configured(),
        "rebrickable_validation_enabled": validate,
        "external_catalog_api_used": "rebrickable",
        "selected_build_strategy": get_build_strategy(analysis),
        "allowed_parts_used": get_allowed_parts_for_analysis(analysis),
        "candidate_parts_catalog": build_candidate_parts_catalog(analysis, data),
        "subject": analysis.get("subject"),
        "category": analysis.get("category"),
        "scene_type": analysis.get("scene_type"),
        "recommended_model_type": analysis.get("recommended_model_type"),
        "build_strategy": analysis.get("build_strategy"),
        "automation_level": analysis.get("automation_level"),
        "recommended_pricing_tier": analysis.get("recommended_pricing_tier"),
        "product_title": analysis.get("product_title"),
        "short_product_description": analysis.get("short_product_description"),
        "build_modules": build_modules,
        "parts": parts,
        "parts_summary": build_parts_summary(parts),
        "estimated_total_parts": sum(p["quantity"] for p in parts),
        "rebrickable_parts_export": create_rebrickable_parts_export(parts),
        "source_analysis": analysis
    }

    if include_export:
        response["basic_parts_xml_export"] = create_basic_xml_export(parts)

    return response


def generate_architecture_plan_from_analysis(analysis, data):
    parts = []
    m = get_tier_multiplier(analysis)

    add_part_line(parts, "Plate 2x2", "green", 40 * m, analysis, "base_landscape", "lawn base")
    add_part_line(parts, "Plate 1x2", "dark_green", 30 * m, analysis, "base_landscape", "hedges")
    add_part_line(parts, "Plant Flower 5 Petals", "red", 8 * m, analysis, "base_landscape", "flower details")

    add_part_line(parts, "Brick Round 2x2", "dark_bluish_gray", 90 * m, analysis, "round_towers", "round tower structure")
    add_part_line(parts, "Brick 1x2", "dark_bluish_gray", 180 * m, analysis, "castle_walls", "stone wall sections")
    add_part_line(parts, "Brick 1x4", "dark_bluish_gray", 80 * m, analysis, "castle_walls", "long wall sections")
    add_part_line(parts, "Masonry Brick 1x2", "dark_bluish_gray", 50 * m, analysis, "castle_walls", "stone texture")
    add_part_line(parts, "Plate 1x1", "black", 28 * m, analysis, "feature_details", "small window openings")

    add_part_line(parts, "Arch 1x4", "tan", 8 * m, analysis, "central_arch", "arch structure")
    add_part_line(parts, "Plate 1x2", "tan", 60 * m, analysis, "central_arch", "facade layers")
    add_part_line(parts, "Tile 1x2", "white", 30 * m, analysis, "central_arch", "light facade highlights")
    add_part_line(parts, "Headlight Brick", "tan", 20 * m, analysis, "feature_details", "offset facade details")

    build_modules = [
        {
            "module_id": "base_landscape",
            "name": "Landscape Base",
            "description": "Green lawn base with hedges and small flower details."
        },
        {
            "module_id": "round_towers",
            "name": "Round Castle Towers",
            "description": "Dark stone cylindrical towers with crenellated tops and small window openings."
        },
        {
            "module_id": "castle_walls",
            "name": "Castle Wall Sections",
            "description": "Dark stone walls connecting the towers with varied grey texture."
        },
        {
            "module_id": "central_arch",
            "name": "Central Arch or Facade",
            "description": "Light-colored arch or facade with layered decorative details."
        },
        {
            "module_id": "feature_details",
            "name": "Small Architectural Details",
            "description": "Small windows, contrast marks, offsets, and ornamental accents."
        }
    ]

    return make_response(
        "Analysis-based architecture brick plan generated",
        analysis,
        build_modules,
        parts,
        data
    )


def generate_mosaic_plan_from_analysis(analysis, data):
    parts = []
    m = get_tier_multiplier(analysis)
    eye_color = choose_eye_color(analysis)

    add_part_line(parts, "Tile 2x2", "tan", 80 * m, analysis, "main_color_zones", "large light color zones")
    add_part_line(parts, "Tile 1x2", "reddish_brown", 90 * m, analysis, "main_color_zones", "medium/dark color zones")
    add_part_line(parts, "Tile 1x1", "black", 35 * m, analysis, "feature_details", "dark outlines and pupils")

    add_part_line(parts, "Tile 1x1", eye_color, 20 * m, analysis, "feature_details", "eye color detail")
    add_part_line(parts, "Tile Round 1x1", eye_color, 12 * m, analysis, "feature_details", "round eye highlights")

    add_part_line(parts, "Tile 1x1", "white", 60 * m, analysis, "feature_details", "white highlights")
    add_part_line(parts, "Plate Round 1x1 Solid Stud", "light_tan", 6 * m, analysis, "feature_details", "nose or small warm detail")
    add_part_line(parts, "Plate 1x2", "light_bluish_gray", 80 * m, analysis, "base_grid", "neutral background/base grid")

    build_modules = [
        {
            "module_id": "base_grid",
            "name": "Mosaic Relief Base",
            "description": "Flat rectangular base matching the image crop."
        },
        {
            "module_id": "main_color_zones",
            "name": "Main Color Zones",
            "description": "Large color-blocked tile areas for the main subject."
        },
        {
            "module_id": "feature_details",
            "name": "Recognizable Feature Details",
            "description": "Small high-contrast areas for eyes, markings, outlines, and expression."
        },
        {
            "module_id": "shallow_relief",
            "name": "Shallow Relief Layers",
            "description": "Small height differences to emphasize key facial or subject features."
        }
    ]

    return make_response(
        "Analysis-based mosaic relief brick plan generated",
        analysis,
        build_modules,
        parts,
        data
    )


def generate_pet_plan_from_analysis(analysis, data):
    parts = []
    m = get_tier_multiplier(analysis)
    eye_color = choose_eye_color(analysis)

    add_part_line(parts, "Tile 2x2", "tan", 60 * m, analysis, "main_body", "main fur color zones")
    add_part_line(parts, "Tile 1x2", "reddish_brown", 80 * m, analysis, "main_body", "fur markings and stripes")
    add_part_line(parts, "Tile 1x1", "black", 35 * m, analysis, "feature_details", "pupils and dark markings")
    add_part_line(parts, "Tile Round 1x1", eye_color, 16 * m, analysis, "feature_details", "round eye color details")
    add_part_line(parts, "Tile 1x1", "white", 50 * m, analysis, "feature_details", "paws, muzzle, and highlights")
    add_part_line(parts, "Plate Round 1x1 Solid Stud", "light_tan", 6 * m, analysis, "feature_details", "nose detail")

    add_part_line(parts, "Brick 1x1", "tan", 25 * m, analysis, "shallow_relief", "small relief build-up")
    add_part_line(parts, "Brick 1x2", "tan", 30 * m, analysis, "shallow_relief", "body/head relief layers")
    add_part_line(parts, "Slope Cheese 1x1", "tan", 20 * m, analysis, "shallow_relief", "soft organic shaping")
    add_part_line(parts, "Slope Curved 2x1", "tan", 12 * m, analysis, "shallow_relief", "rounded organic contours")

    build_modules = [
        {
            "module_id": "base_grid",
            "name": "Pet Relief Base",
            "description": "Flat base matching the image crop and subject pose."
        },
        {
            "module_id": "main_body",
            "name": "Main Fur and Body Zones",
            "description": "Large color zones representing fur, body outline, and major markings."
        },
        {
            "module_id": "feature_details",
            "name": "Face and Expression Details",
            "description": "Eyes, nose, mouth, paws, stripes, highlights, and other recognizable details."
        },
        {
            "module_id": "shallow_relief",
            "name": "Organic Relief Layers",
            "description": "Small 3D layers for face, ears, body, and soft curves."
        }
    ]

    return make_response(
        "Analysis-based pet brick plan generated",
        analysis,
        build_modules,
        parts,
        data
    )


def generate_portrait_plan_from_analysis(analysis, data):
    parts = []
    m = get_tier_multiplier(analysis)
    eye_color = choose_eye_color(analysis)

    add_part_line(parts, "Tile 2x2", "tan", 70 * m, analysis, "face_base", "main face color zones")
    add_part_line(parts, "Tile 1x2", "reddish_brown", 50 * m, analysis, "hair_shadow", "hair or shadow zones")
    add_part_line(parts, "Tile 1x1", "black", 40 * m, analysis, "feature_details", "eyes, pupils, outlines")
    add_part_line(parts, "Tile Round 1x1", eye_color, 10 * m, analysis, "feature_details", "eye color detail")
    add_part_line(parts, "Tile 1x1", "white", 35 * m, analysis, "feature_details", "eye highlights and light details")
    add_part_line(parts, "Plate 1x2", "light_bluish_gray", 60 * m, analysis, "base_grid", "background/base grid")
    add_part_line(parts, "Slope Cheese 1x1", "tan", 20 * m, analysis, "shallow_relief", "nose/face shaping")
    add_part_line(parts, "Slope Curved 2x1", "tan", 14 * m, analysis, "shallow_relief", "soft facial contours")

    build_modules = [
        {
            "module_id": "base_grid",
            "name": "Portrait Base",
            "description": "Flat or shallow-relief base matching the portrait crop."
        },
        {
            "module_id": "face_base",
            "name": "Face Color Zones",
            "description": "Main skin-tone and face-shape regions."
        },
        {
            "module_id": "hair_shadow",
            "name": "Hair and Shadow Zones",
            "description": "Darker areas for hair, eyebrows, outlines, and shadows."
        },
        {
            "module_id": "feature_details",
            "name": "Facial Details",
            "description": "Eyes, nose, mouth, highlights, and key expression features."
        },
        {
            "module_id": "shallow_relief",
            "name": "Shallow Portrait Relief",
            "description": "Small depth changes to suggest facial structure."
        }
    ]

    return make_response(
        "Analysis-based portrait brick plan generated",
        analysis,
        build_modules,
        parts,
        data
    )


def generate_vehicle_plan_from_analysis(analysis, data):
    parts = []
    m = get_tier_multiplier(analysis)
    primary_color = choose_primary_color(analysis, fallback="light_bluish_gray")

    add_part_line(parts, "Plate 2x4", "black", 20 * m, analysis, "chassis", "vehicle base")
    add_part_line(parts, "Plate 2x6", "black", 12 * m, analysis, "chassis", "long chassis sections")
    add_part_line(parts, "Brick 1x2", primary_color, 80 * m, analysis, "body_shell", "main body shell")
    add_part_line(parts, "Brick 1x4", primary_color, 40 * m, analysis, "body_shell", "long body sections")
    add_part_line(parts, "Tile 1x2", "black", 30 * m, analysis, "windows_lights", "windows and dark details")
    add_part_line(parts, "Tile 1x4", "black", 14 * m, analysis, "windows_lights", "long window areas")
    add_part_line(parts, "Slope 1x2", primary_color, 24 * m, analysis, "body_shell", "sloped body shaping")
    add_part_line(parts, "Slope Curved 2x1", primary_color, 20 * m, analysis, "body_shell", "curved vehicle body shaping")
    add_part_line(parts, "Plate 1x1", "red", 8 * m, analysis, "windows_lights", "rear lights or small accents")

    build_modules = [
        {
            "module_id": "chassis",
            "name": "Vehicle Chassis",
            "description": "Stable rectangular base for the vehicle."
        },
        {
            "module_id": "body_shell",
            "name": "Body Shell",
            "description": "Main color-blocked vehicle body shape."
        },
        {
            "module_id": "windows_lights",
            "name": "Windows and Lights",
            "description": "Dark window zones and small front/rear light details."
        }
    ]

    return make_response(
        "Analysis-based vehicle brick plan generated",
        analysis,
        build_modules,
        parts,
        data
    )


def generate_landscape_plan_from_analysis(analysis, data):
    parts = []
    m = get_tier_multiplier(analysis)

    add_part_line(parts, "Plate 2x4", "green", 80 * m, analysis, "terrain_base", "grass/terrain base")
    add_part_line(parts, "Plate 2x6", "green", 40 * m, analysis, "terrain_base", "larger terrain base")
    add_part_line(parts, "Tile 2x2", "blue", 30 * m, analysis, "water_sky", "water or sky color zone")
    add_part_line(parts, "Tile 1x2", "light_bluish_gray", 40 * m, analysis, "rocks_paths", "rocks, path, or neutral scenery")
    add_part_line(parts, "Slope 1x2", "dark_bluish_gray", 24 * m, analysis, "rocks_paths", "rocky slopes")
    add_part_line(parts, "Slope Curved 2x1", "green", 20 * m, analysis, "terrain_base", "soft terrain contours")
    add_part_line(parts, "Plant Leaves 4x3", "green", 10 * m, analysis, "plants", "leaf clusters")
    add_part_line(parts, "Plant 1x1 Round 3 Leaves", "green", 16 * m, analysis, "plants", "small plants")
    add_part_line(parts, "Plant Flower 5 Petals", "red", 8 * m, analysis, "plants", "flower accents")

    build_modules = [
        {
            "module_id": "terrain_base",
            "name": "Terrain Base",
            "description": "Layered ground base for landscape or outdoor scenery."
        },
        {
            "module_id": "water_sky",
            "name": "Water or Sky Color Zones",
            "description": "Blue scenic regions used when water or sky is part of the build."
        },
        {
            "module_id": "rocks_paths",
            "name": "Rocks and Paths",
            "description": "Neutral slopes and tiles for rocks, paths, or mountains."
        },
        {
            "module_id": "plants",
            "name": "Plants and Natural Details",
            "description": "Leaves, flower elements, and vegetation accents."
        }
    ]

    return make_response(
        "Analysis-based landscape brick plan generated",
        analysis,
        build_modules,
        parts,
        data
    )


def generate_generic_plan_from_analysis(analysis, data):
    parts = []
    m = get_tier_multiplier(analysis)
    primary_color = choose_primary_color(analysis, fallback="light_bluish_gray")

    add_part_line(parts, "Plate 2x2", primary_color, 40 * m, analysis, "display_base", "simple display base")
    add_part_line(parts, "Brick 1x2", primary_color, 80 * m, analysis, "main_subject", "main subject body")
    add_part_line(parts, "Brick 1x4", primary_color, 30 * m, analysis, "main_subject", "long body sections")
    add_part_line(parts, "Plate 1x2", "dark_bluish_gray", 40 * m, analysis, "main_subject", "shadow and detail layers")
    add_part_line(parts, "Tile 1x2", "tan", 30 * m, analysis, "surface_detail", "surface color detail")
    add_part_line(parts, "Slope Cheese 1x1", primary_color, 14 * m, analysis, "surface_detail", "small shape details")
    add_part_line(parts, "Tile Round 1x1", "black", 8 * m, analysis, "feature_details", "small circular details")

    build_modules = [
        {
            "module_id": "display_base",
            "name": "Display Base",
            "description": "Simple base for presenting the model."
        },
        {
            "module_id": "main_subject",
            "name": "Main Subject",
            "description": "Template-based brick representation of the analyzed subject."
        },
        {
            "module_id": "surface_detail",
            "name": "Surface Details",
            "description": "Small tiles and slopes for surface color and shape details."
        },
        {
            "module_id": "feature_details",
            "name": "Feature Details",
            "description": "Small recognisable accents and contrast details."
        }
    ]

    return make_response(
        "Generic analysis-based brick plan generated",
        analysis,
        build_modules,
        parts,
        data
    )



# ---------------------------------------------------------------------
# Image-to-placement geometry helpers
# ---------------------------------------------------------------------

ARCHITECTURE_LEGO_RGB_PALETTE = {
    # Core neutrals
    "black": (5, 19, 29),
    "white": (255, 255, 255),

    # Stone / castle grays
    "very_light_gray": (229, 228, 222),
    "light_gray": (155, 161, 157),
    "light_bluish_gray": (160, 165, 169),
    "medium_gray": (128, 128, 128),
    "dark_gray": (109, 110, 108),
    "dark_bluish_gray": (99, 95, 98),

    # Sand / tan / masonry colors
    "light_tan": (238, 229, 195),
    "tan": (215, 197, 153),
    "dark_tan": (149, 138, 115),
    "sand_yellow": (215, 197, 153),
    "nougat": (204, 142, 104),
    "medium_nougat": (170, 125, 85),

    # Browns / shadows
    "dark_brown": (53, 33, 0),
    "brown": (96, 57, 19),
    "reddish_brown": (88, 42, 18),

    # Greens / landscape
    "sand_green": (120, 144, 129),
    "dark_green": (24, 70, 50),
    "green": (40, 127, 70),
    "lime": (187, 233, 11),

    # Blues / sky
    "sand_blue": (96, 116, 161),
    "light_blue": (180, 210, 227),
    "medium_blue": (90, 147, 219),
    "blue": (13, 105, 171),

    # Warm accent colors
    "dark_red": (123, 46, 47),
    "red": (196, 40, 28),
    "orange": (218, 133, 64),
    "yellow": (245, 205, 47),
    "pink": (255, 167, 176)
}


PET_LEGO_RGB_PALETTE = {
    # Pet-safe palette for realistic fur/portrait mosaics.
    # Red/orange/pink/yellow are intentionally removed to avoid fake warm patches.
    # Green is also removed from the general fur palette; eye greens are handled separately.
    "black": (5, 19, 29),
    "white": (255, 255, 255),

    # Neutral fur / background / shadow colors
    "very_light_gray": (229, 228, 222),
    "light_gray": (155, 161, 157),
    "light_bluish_gray": (160, 165, 169),
    "medium_gray": (128, 128, 128),
    "dark_gray": (109, 110, 108),
    "dark_bluish_gray": (99, 95, 98),

    # Cream / tan fur colors, locally biased warmer for cat/dog fur mapping
    "light_tan": (232, 214, 176),
    "tan": (205, 178, 128),
    "dark_tan": (154, 128, 91),

    # Brown fur / stripe colors
    "dark_brown": (53, 33, 0),
    "brown": (105, 70, 40),
    "reddish_brown": (95, 50, 25)
}


PET_EYE_RGB_PALETTE = {
    # Eye-only colors for cats/dogs.
    # These are not used for the body/background, only for greenish/hazel eye pixels.
    "sand_green": (120, 144, 129),
    "olive_green": (128, 128, 48),
    "dark_green": (24, 70, 50),
    "green": (40, 127, 70),
    "lime": (187, 233, 11),
    "black": (5, 19, 29),
    "white": (255, 255, 255)
}


def get_palette_for_analysis(analysis):
    """
    Uses different color palettes by subject type.
    This prevents pet images from picking sky-blue colors while still allowing
    architecture/castle images to use stone, sand, and sky colors.
    """
    category = str(analysis.get("category", "")).lower()
    model_type = str(analysis.get("recommended_model_type", "")).lower()
    strategy = str(analysis.get("build_strategy", "")).lower()

    if (
        category == "pet_animal"
        or "pet" in category
        or "animal" in category
        or "pet" in strategy
        or "portrait" in model_type
    ):
        return PET_LEGO_RGB_PALETTE

    return ARCHITECTURE_LEGO_RGB_PALETTE

def rgb_tuple_to_hex(rgb):
    if not rgb:
        return None

    r, g, b = rgb
    return f"{int(r):02X}{int(g):02X}{int(b):02X}"



def clamp_int(value, minimum, maximum, default):
    try:
        value = int(value)
    except Exception:
        return default

    return max(minimum, min(maximum, value))


def download_image_from_url(image_url):
    if not image_url:
        raise ValueError("Missing original_image_url.")

    response = requests.get(image_url, timeout=30)
    response.raise_for_status()

    return Image.open(BytesIO(response.content)).convert("RGB")


def resize_image_keep_aspect(image, width, height):
    """
    Makes a square/rectangular canvas without stretching the source image.
    This keeps the uploaded image shape more natural for mosaic geometry.
    """
    image.thumbnail((width, height))

    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    offset_x = (width - image.width) // 2
    offset_y = (height - image.height) // 2

    canvas.paste(image, (offset_x, offset_y))
    return canvas


def closest_lego_color_name(rgb, palette):
    r, g, b = rgb

    best_name = "light_bluish_gray"
    best_distance = None

    for color_name, color_rgb in palette.items():
        cr, cg, cb = color_rgb

        distance = (
            (r - cr) ** 2 +
            (g - cg) ** 2 +
            (b - cb) ** 2
        )

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_name = color_name

    return best_name


def is_pet_like_analysis(analysis):
    """
    Returns True for pet/animal inputs where special fur/eye color handling is needed.
    """
    category = str(analysis.get("category", "")).lower()
    strategy = str(analysis.get("build_strategy", "")).lower()
    model_type = str(analysis.get("recommended_model_type", "")).lower()
    subject = str(analysis.get("subject", "")).lower()

    return (
        category == "pet_animal"
        or "pet" in category
        or "animal" in category
        or "pet" in strategy
        or model_type == "mosaic_relief"
        or "cat" in subject
        or "kitten" in subject
        or "dog" in subject
        or "puppy" in subject
    )


def is_pet_eye_greenish_pixel(rgb):
    """
    Detects real cat/dog eye-like green/hazel pixels.
    Cat eyes are often pale gray-green, yellow-green, or olive-green, not pure green.
    This is intentionally broader than simple green dominance, but still avoids
    dark body/background shadows becoming green.
    """
    r, g, b = rgb
    brightness = brightness_from_rgb(rgb)
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    saturation = max_c - min_c

    if brightness < 45 or brightness > 215:
        return False

    # Pure or clear green/gray-green pixels.
    greenish = (
        g >= r - 14
        and g >= b - 6
        and saturation >= 16
    )

    # Hazel/olive eye tones: red and green are close, blue is lower.
    olive_or_hazel = (
        g >= b + 8
        and r >= b + 6
        and abs(r - g) <= 52
        and saturation >= 18
        and brightness >= 55
        and brightness <= 195
    )

    return bool(greenish or olive_or_hazel)


def closest_pet_eye_color_name(rgb):
    """
    Maps an eye-like pixel to the nearest eye-only LEGO/Rebrickable color.
    This protects the iris while keeping green out of fur and shadows.
    """
    return closest_lego_color_name(rgb, PET_EYE_RGB_PALETTE)


def remap_pet_unwanted_warm_colors(color_name, rgb):
    """
    Prevents pet mosaics from using bright red/orange/pink/yellow pixels.
    These colors usually come from warm fur, nose areas, lighting, compression,
    or edge artifacts. For pet mosaics they should become natural fur colors.
    """
    blocked = {"red", "dark_red", "orange", "pink", "yellow"}

    if color_name not in blocked:
        return color_name

    r, g, b = rgb
    brightness = brightness_from_rgb(rgb)

    if brightness < 70:
        return "dark_brown"

    if r > g and g >= b:
        if brightness < 115:
            return "reddish_brown"
        if brightness < 165:
            return "brown"
        return "dark_tan"

    if brightness >= 185:
        return "light_tan"

    return "tan"


def correct_pet_fur_color_mapping(color_name, rgb):
    """
    Makes pet fur colors more natural:
    - keeps green out of fur/body/background
    - maps warm fur toward tan/brown instead of white/gray
    - keeps truly bright background/highlights light
    """
    r, g, b = rgb
    brightness = brightness_from_rgb(rgb)

    # Safety: if any green was selected outside eye handling, remap it to fur colors.
    if color_name in {"sand_green", "olive_green", "dark_green", "green", "lime"}:
        if brightness < 80:
            return "dark_brown"
        if brightness < 125:
            return "brown"
        if brightness < 170:
            return "dark_tan"
        return "tan"

    color_name = remap_pet_unwanted_warm_colors(color_name, rgb)

    # Warm fur: red >= green >= blue or red clearly above blue.
    warm_pixel = (r >= g - 8 and g >= b - 8 and (r - b) > 16)

    if warm_pixel:
        if brightness > 232:
            return "light_tan"
        if brightness > 188:
            return "tan"
        if brightness > 138:
            return "dark_tan"
        if brightness > 82:
            return "brown"
        return "dark_brown"

    # Avoid over-whitening the cat body. Very bright pixels can stay white, but
    # mid-bright fur should be light tan instead of white/light gray.
    if color_name == "white" and brightness < 235:
        return "light_tan"

    if color_name in {"very_light_gray", "light_gray", "light_bluish_gray"}:
        # Slightly warm or beige pixels should become tan family.
        if r >= b + 10 and g >= b + 4 and brightness < 225:
            if brightness > 185:
                return "light_tan"
            return "tan"

    return color_name


def get_geometry_palette_for_resolution(base_palette, analysis):
    """
    Returns palette used for resolving preview RGB. For pet geometry, include
    eye-only colors so sand_green/olive_green/green can resolve correctly.
    """
    if is_pet_like_analysis(analysis):
        merged = dict(base_palette)
        merged.update(PET_EYE_RGB_PALETTE)
        return merged
    return base_palette

def brightness_from_rgb(rgb):
    r, g, b = rgb
    return (0.299 * r) + (0.587 * g) + (0.114 * b)


def height_plates_from_rgb(rgb, edge_value=0):
    """
    Basic relief height:
    - darker pixels become slightly lower
    - brighter pixels become slightly higher
    - edge pixels get a small boost so facial/details are more visible
    """
    brightness = brightness_from_rgb(rgb)

    if brightness < 55:
        height = 1
    elif brightness < 100:
        height = 2
    elif brightness < 150:
        height = 3
    elif brightness < 205:
        height = 4
    else:
        height = 5

    if edge_value > 35:
        height += 1

    return max(1, min(6, height))


def resolve_geometry_color(color_name, palette):
    """
    Uses Rebrickable color data when available.
    If a custom preview color is not found in Rebrickable, it falls back to
    the selected local palette RGB so the preview does not turn gray.
    """
    palette_rgb_hex = rgb_tuple_to_hex(palette.get(color_name))

    try:
        color_data = resolve_rebrickable_color(color_name)
        resolved_rgb = color_data.get("rgb") or palette_rgb_hex

        return {
            "planner_color": color_name,
            "rebrickable_color_id": color_data.get("id"),
            "rebrickable_color_name": color_data.get("name") or color_name,
            "rebrickable_color_rgb": resolved_rgb
        }

    except Exception:
        return {
            "planner_color": color_name,
            "rebrickable_color_id": None,
            "rebrickable_color_name": color_name,
            "rebrickable_color_rgb": palette_rgb_hex
        }


def generate_image_geometry(image_url, analysis, width=48, height=48, part_name="Plate 1x1"):
    """
    Converts the original uploaded image into real grid placement geometry.
    Each placement has x, y, z, color, part number, and relief height.
    This is a practical first geometry layer for mosaic/shallow relief output.
    """
    width = clamp_int(width, 8, 256, 48)
    height = clamp_int(height, 8, 256, 48)

    image = download_image_from_url(image_url)
    image = resize_image_keep_aspect(image, width, height)

    palette = get_palette_for_analysis(analysis)

    edge_image = image.convert("L").filter(ImageFilter.FIND_EDGES)

    part_num = "3024"

    placements = []
    color_cache = {}

    for y in range(height):
        for x in range(width):
            rgb = image.getpixel((x, y))
            edge_value = edge_image.getpixel((x, y))

            color_name = closest_lego_color_name(rgb, palette)

            # Pet/animal images need special handling:
            # - green/olive/sand-green only for genuine eye-like pixels
            # - warm fur should stay tan/brown, not white/gray/green/red
            if is_pet_like_analysis(analysis):
                if is_pet_eye_greenish_pixel(rgb):
                    color_name = closest_pet_eye_color_name(rgb)
                else:
                    color_name = correct_pet_fur_color_mapping(color_name, rgb)

            if color_name not in color_cache:
                resolution_palette = get_geometry_palette_for_resolution(palette, analysis)
                color_cache[color_name] = resolve_geometry_color(color_name, resolution_palette)

            color_data = color_cache[color_name]
            height_plates = height_plates_from_rgb(rgb, edge_value=edge_value)

            placements.append({
                "x": x,
                "y": y,
                "z": 0,
                "part_name": part_name,
                "part_num": part_num,
                "color": color_name,
                "rebrickable_color_id": color_data.get("rebrickable_color_id"),
                "rebrickable_color_name": color_data.get("rebrickable_color_name"),
                "rebrickable_color_rgb": color_data.get("rebrickable_color_rgb"),
                "height_plates": height_plates,
                "orientation": 0,
                "source_rgb": {
                    "r": rgb[0],
                    "g": rgb[1],
                    "b": rgb[2]
                }
            })

    return {
        "type": "image_pixel_mosaic_relief_geometry",
        "source": "original_image_url",
        "width": width,
        "height": height,
        "stud_count": width * height,
        "part_name": part_name,
        "part_num": part_num,
        "max_height_plates": 6,
        "color_policy": {
            "palette_type": "pet_safe_warm_fur_eye_protected" if is_pet_like_analysis(analysis) else "general",
            "pet_unwanted_warm_colors_blocked": bool(is_pet_like_analysis(analysis)),
            "pet_warm_fur_bias_enabled": bool(is_pet_like_analysis(analysis)),
            "pet_green_limited_to_eye_pixels": bool(is_pet_like_analysis(analysis)),
            "eye_color_palette": list(PET_EYE_RGB_PALETTE.keys()) if is_pet_like_analysis(analysis) else [],
            "blocked_for_pet_geometry": ["red", "dark_red", "orange", "pink", "yellow"] if is_pet_like_analysis(analysis) else []
        },
        "placements": placements
    }


def summarize_geometry_parts(geometry):
    summary = {}

    for placement in geometry.get("placements", []):
        color_name = placement.get("rebrickable_color_name") or placement.get("color")
        key = f"{placement.get('part_name')} - {color_name}"
        summary[key] = summary.get(key, 0) + 1

    return summary


def hex_to_rgb(hex_value, fallback=(160, 165, 169)):
    if not hex_value:
        return fallback

    try:
        hex_value = str(hex_value).replace("#", "").strip()

        if len(hex_value) != 6:
            return fallback

        return (
            int(hex_value[0:2], 16),
            int(hex_value[2:4], 16),
            int(hex_value[4:6], 16)
        )

    except Exception:
        return fallback


def darken_rgb(rgb, amount=35):
    r, g, b = rgb
    return (
        max(0, r - amount),
        max(0, g - amount),
        max(0, b - amount)
    )


def lighten_rgb(rgb, amount=35):
    r, g, b = rgb
    return (
        min(255, r + amount),
        min(255, g + amount),
        min(255, b + amount)
    )


def create_stud_preview_base64(geometry, cell_size=14):
    """
    Creates a 2D LEGO-style stud preview from image_geometry.
    Returns PNG as base64 string.
    """
    width = int(geometry.get("width", 48))
    height = int(geometry.get("height", 48))

    cell_size = clamp_int(cell_size, 3, 30, 6)

    canvas_width = width * cell_size
    canvas_height = height * cell_size

    image = Image.new("RGB", (canvas_width, canvas_height), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    placements = geometry.get("placements", [])

    for placement in placements:
        x = int(placement.get("x", 0))
        y = int(placement.get("y", 0))

        px = x * cell_size
        py = y * cell_size

        color_hex = placement.get("rebrickable_color_rgb")

        if color_hex:
            color_rgb = hex_to_rgb(color_hex)
        else:
            # Fallback to the original image pixel color.
            # This prevents the preview from becoming grey when Rebrickable RGB is missing.
            source_rgb = placement.get("source_rgb", {})
            color_rgb = (
                int(source_rgb.get("r", 160)),
                int(source_rgb.get("g", 165)),
                int(source_rgb.get("b", 169))
            )

        height_plates = int(placement.get("height_plates", 1))

        # Slightly adjust the color by height so shallow-relief depth is visible.
        if height_plates >= 5:
            fill_rgb = lighten_rgb(color_rgb, 15)
        elif height_plates <= 2:
            fill_rgb = darken_rgb(color_rgb, 15)
        else:
            fill_rgb = color_rgb

        outline_rgb = darken_rgb(fill_rgb, 45)
        highlight_rgb = lighten_rgb(fill_rgb, 35)

        # Main square tile/plate cell.
        draw.rectangle(
            [px, py, px + cell_size - 1, py + cell_size - 1],
            fill=fill_rgb,
            outline=outline_rgb
        )

        # Round stud on top.
        margin = max(2, cell_size // 4)
        stud_box = [
            px + margin,
            py + margin,
            px + cell_size - margin,
            py + cell_size - margin
        ]

        draw.ellipse(
            stud_box,
            fill=lighten_rgb(fill_rgb, 18),
            outline=outline_rgb
        )

        # Small highlight dot for a more LEGO-like look.
        dot_size = max(1, cell_size // 7)
        draw.ellipse(
            [
                px + margin + 1,
                py + margin + 1,
                px + margin + 1 + dot_size,
                py + margin + 1 + dot_size
            ],
            fill=highlight_rgb
        )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_from_analysis(data):
    analysis = data.get("analysis", {})

    model_type = analysis.get("recommended_model_type", "")
    category = analysis.get("category", "")
    strategy = get_build_strategy(analysis)

    if model_type == "mosaic_relief" or strategy == "mosaic_or_relief_conversion":
        return generate_mosaic_plan_from_analysis(analysis, data)

    if strategy == "pet_template_customization" or category == "pet_animal":
        return generate_pet_plan_from_analysis(analysis, data)

    if strategy == "portrait_bust_template" or category == "person_portrait":
        return generate_portrait_plan_from_analysis(analysis, data)

    if model_type in ["architecture_full_model", "architecture_facade"] or strategy == "architecture_studio_rebuild":
        return generate_architecture_plan_from_analysis(analysis, data)

    if model_type == "vehicle_model" or strategy == "vehicle_template_customization" or category == "vehicle":
        return generate_vehicle_plan_from_analysis(analysis, data)

    if strategy == "landscape_diorama_rebuild" or category == "landscape_scene":
        return generate_landscape_plan_from_analysis(analysis, data)

    return generate_generic_plan_from_analysis(analysis, data)


@app.post("/generate-lego-model")
async def generate_lego_model(data: dict):
    analysis = data.get("analysis")

    if not analysis:
        return {
            "error": "Missing analysis JSON",
            "expected_body": {
                "analysis": {},
                "original_image_url": "https://...",
                "original_image_object_name": "uploads/user-id/file.jpg",
                "include_image_geometry": True,
                "geometry_width": 48,
                "geometry_height": 48,
                "include_preview_image": True,
                "preview_cell_size": 6,
                "include_basic_xml_export": True,
                "include_build_modules": True,
                "detail_level": 2,
                "validate_with_rebrickable": True,
                "include_catalog_discovery": True,
                "preferred_surface_style": "smooth_tiles"
            }
        }

    response = generate_from_analysis(data)

    original_image_url = data.get("original_image_url")
    include_image_geometry = bool(data.get("include_image_geometry", False))

    response["original_image_url"] = original_image_url
    response["original_image_object_name"] = data.get("original_image_object_name")
    response["include_image_geometry"] = include_image_geometry

    if include_image_geometry:
        if not original_image_url:
            response["image_geometry_error"] = "include_image_geometry=true but original_image_url is missing."
            return response

        try:
            geometry_width = data.get("geometry_width", 48)
            geometry_height = data.get("geometry_height", 48)

            image_geometry = generate_image_geometry(
                original_image_url,
                analysis,
                width=geometry_width,
                height=geometry_height,
                part_name="Plate 1x1"
            )

            response["generation_mode"] = "analysis_plus_image_geometry"
            response["image_geometry"] = image_geometry
            response["image_geometry_parts_summary"] = summarize_geometry_parts(image_geometry)
            response["image_geometry_estimated_total_parts"] = len(image_geometry.get("placements", []))

            include_preview_image = bool(data.get("include_preview_image", True))

            if include_preview_image:
                preview_cell_size = data.get("preview_cell_size", 6)

                response["preview_image_base64"] = create_stud_preview_base64(
                    image_geometry,
                    cell_size=preview_cell_size
                )
                response["preview_image_format"] = "png"
                response["preview_image_type"] = "2d_stud_preview"

        except Exception as e:
            response["generation_mode"] = "analysis_only_geometry_failed"
            response["image_geometry_error"] = str(e)

    return response
