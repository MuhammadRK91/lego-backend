from fastapi import FastAPI
import os
import re
import requests
import base64
from functools import lru_cache
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw

app = FastAPI()

REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3"


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
        "catalog_source": "rebrickable",
        "rebrickable_api_configured": rebrickable_is_configured(),
        "external_catalog_api_used": "rebrickable",
        "note": "Colors are resolved through Rebrickable API. Parts are selected from strategy-specific controlled libraries and validated with Rebrickable. If original_image_url is sent with include_image_geometry=true, the server creates image-based placement geometry. If include_preview_image=true, the server returns a 2D stud preview as base64 PNG. If include_clean_mosaic_preview=true, it also returns a cleaner square-tile preview without studs."
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

        "dark green": "dark green"
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
    add_part_line(parts, "Plate Round 1x1 Solid Stud", "red", 6 * m, analysis, "feature_details", "nose or small warm detail")
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
    add_part_line(parts, "Plate Round 1x1 Solid Stud", "red", 6 * m, analysis, "feature_details", "nose detail")

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

LEGO_RGB_PALETTE = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (196, 40, 28),
    "blue": (13, 105, 171),
    "green": (40, 127, 70),
    "dark_green": (24, 70, 50),
    "yellow": (245, 205, 47),
    "orange": (218, 133, 64),
    "tan": (215, 197, 153),
    "reddish_brown": (88, 42, 18),
    "light_bluish_gray": (160, 165, 169),
    "dark_bluish_gray": (99, 95, 98),
    "lime": (187, 233, 11),
    "medium_blue": (90, 147, 219),
    "pink": (255, 167, 176)
}


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


def closest_lego_color_name(rgb):
    r, g, b = rgb

    best_name = "light_bluish_gray"
    best_distance = None

    for color_name, color_rgb in LEGO_RGB_PALETTE.items():
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


def resolve_geometry_color(color_name):
    """
    Uses your existing Rebrickable color resolver when available.
    If Rebrickable is not configured, geometry still works with color names.
    """
    try:
        color_data = resolve_rebrickable_color(color_name)

        return {
            "planner_color": color_name,
            "rebrickable_color_id": color_data.get("id"),
            "rebrickable_color_name": color_data.get("name"),
            "rebrickable_color_rgb": color_data.get("rgb")
        }

    except Exception:
        return {
            "planner_color": color_name,
            "rebrickable_color_id": None,
            "rebrickable_color_name": color_name,
            "rebrickable_color_rgb": None
        }


def generate_image_geometry(image_url, width=48, height=48, part_name="Plate 1x1"):
    """
    Converts the original uploaded image into real grid placement geometry.
    Each placement has x, y, z, color, part number, and relief height.
    This is a practical first geometry layer for mosaic/shallow relief output.
    """
    width = clamp_int(width, 8, 128, 48)
    height = clamp_int(height, 8, 128, 48)

    image = download_image_from_url(image_url)
    image = resize_image_keep_aspect(image, width, height)

    edge_image = image.convert("L").filter(ImageFilter.FIND_EDGES)

    part_num = "3024"

    placements = []
    color_cache = {}

    for y in range(height):
        for x in range(width):
            rgb = image.getpixel((x, y))
            edge_value = edge_image.getpixel((x, y))

            color_name = closest_lego_color_name(rgb)

            if color_name not in color_cache:
                color_cache[color_name] = resolve_geometry_color(color_name)

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

    cell_size = clamp_int(cell_size, 6, 30, 14)

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


def create_clean_mosaic_preview_base64(geometry, cell_size=6, draw_grid=False):
    """
    Creates a cleaner square-tile mosaic preview from image_geometry.
    This preview does not draw studs, so photo details are easier to see.
    Returns PNG as base64 string.
    """
    width = int(geometry.get("width", 128))
    height = int(geometry.get("height", 72))

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
            source_rgb = placement.get("source_rgb", {})
            color_rgb = (
                int(source_rgb.get("r", 160)),
                int(source_rgb.get("g", 165)),
                int(source_rgb.get("b", 169))
            )

        height_plates = int(placement.get("height_plates", 1))

        # Keep relief visible but subtle. Strong contrast makes photo previews noisy.
        if height_plates >= 5:
            fill_rgb = lighten_rgb(color_rgb, 6)
        elif height_plates <= 2:
            fill_rgb = darken_rgb(color_rgb, 6)
        else:
            fill_rgb = color_rgb

        if draw_grid:
            outline_rgb = darken_rgb(fill_rgb, 25)
        else:
            outline_rgb = fill_rgb

        draw.rectangle(
            [px, py, px + cell_size - 1, py + cell_size - 1],
            fill=fill_rgb,
            outline=outline_rgb
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
                "include_clean_mosaic_preview": True,
                "clean_preview_cell_size": 6,
                "clean_preview_draw_grid": False,
                "include_basic_xml_export": True,
                "include_build_modules": True,
                "detail_level": 2,
                "validate_with_rebrickable": True
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

            include_clean_mosaic_preview = bool(data.get("include_clean_mosaic_preview", True))

            if include_clean_mosaic_preview:
                clean_preview_cell_size = data.get("clean_preview_cell_size", data.get("preview_cell_size", 6))
                clean_preview_draw_grid = bool(data.get("clean_preview_draw_grid", False))

                response["clean_mosaic_preview_base64"] = create_clean_mosaic_preview_base64(
                    image_geometry,
                    cell_size=clean_preview_cell_size,
                    draw_grid=clean_preview_draw_grid
                )
                response["clean_mosaic_preview_format"] = "png"
                response["clean_mosaic_preview_type"] = "clean_square_tile_preview"

        except Exception as e:
            response["generation_mode"] = "analysis_only_geometry_failed"
            response["image_geometry_error"] = str(e)

    return response
