from fastapi import FastAPI
import os
import re
import requests
from functools import lru_cache

app = FastAPI()

REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3"


# Strategy-specific allowed parts.
# These are controlled part libraries, not blind Rebrickable searches.
# Rebrickable is used to validate these exact part numbers and colors.
ALLOWED_PARTS_BY_STRATEGY = {
    "mosaic_or_relief_conversion": {
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 2x2": "3068b",
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 2x2": "3022"
    },
    "pet_template_customization": {
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 2x2": "3068b",
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 2x2": "3022",
        "Brick 1x1": "3005",
        "Brick 1x2": "3004"
    },
    "architecture_studio_rebuild": {
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 2x2": "3022",
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 2x2": "3068b",
        "Brick 1x1": "3005",
        "Brick 1x2": "3004",
        "Brick 1x4": "3010",
        "Brick Round 2x2": "3941",
        "Arch 1x4": "3659"
    },
    "vehicle_template_customization": {
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 2x2": "3022",
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Brick 1x1": "3005",
        "Brick 1x2": "3004",
        "Brick 1x4": "3010",
        "Slope 1x2": "3040"
    },
    "object_depth_to_voxel": {
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 2x2": "3022",
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Brick 1x1": "3005",
        "Brick 1x2": "3004",
        "Brick 1x4": "3010"
    },
    "landscape_diorama_rebuild": {
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 2x2": "3022",
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b",
        "Tile 2x2": "3068b",
        "Brick 1x1": "3005",
        "Brick 1x2": "3004"
    },
    "manual_review_required": {
        "Plate 1x1": "3024",
        "Plate 1x2": "3023",
        "Plate 2x2": "3022",
        "Tile 1x1": "3070b",
        "Tile 1x2": "3069b"
    }
}


@app.get("/")
def root():
    return {
        "status": "running",
        "mode": "analysis_only",
        "catalog_source": "rebrickable",
        "rebrickable_api_configured": rebrickable_is_configured(),
        "external_catalog_api_used": "rebrickable",
        "note": "Colors are resolved through Rebrickable API. Parts are selected from strategy-specific controlled libraries and validated with Rebrickable."
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
            "is_trans": color.get("is_trans")
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
        "is_trans": False
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

    if category in ["pet_animal", "person_portrait"]:
        return "mosaic_or_relief_conversion"

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
    include_export = bool(data.get("include_wanted_list_xml", True))

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
    add_part_line(parts, "Plate 1x1", "red", 16 * m, analysis, "base_landscape", "flower details")

    add_part_line(parts, "Brick Round 2x2", "dark_bluish_gray", 90 * m, analysis, "round_towers", "round tower structure")
    add_part_line(parts, "Brick 1x2", "dark_bluish_gray", 180 * m, analysis, "castle_walls", "stone wall sections")
    add_part_line(parts, "Brick 1x4", "dark_bluish_gray", 80 * m, analysis, "castle_walls", "long wall sections")
    add_part_line(parts, "Plate 1x1", "black", 28 * m, analysis, "feature_details", "small window openings")

    add_part_line(parts, "Arch 1x4", "tan", 8 * m, analysis, "central_arch", "arch structure")
    add_part_line(parts, "Plate 1x2", "tan", 60 * m, analysis, "central_arch", "facade layers")
    add_part_line(parts, "Tile 1x2", "white", 30 * m, analysis, "central_arch", "light facade highlights")

    build_modules = [
        {
            "module_id": "base_landscape",
            "name": "Landscape Base",
            "description": "Green lawn base with hedges and small red flower details."
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
            "name": "Central Triumphal Arch",
            "description": "Light-colored arch facade with layered decorative details."
        },
        {
            "module_id": "feature_details",
            "name": "Small Architectural Details",
            "description": "Small windows, contrast marks, and ornamental accents."
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

    add_part_line(parts, "Tile 2x2", "tan", 80 * m, analysis, "main_color_zones", "light fur base color")
    add_part_line(parts, "Tile 1x2", "reddish_brown", 90 * m, analysis, "main_color_zones", "tabby stripe and dark fur zones")
    add_part_line(parts, "Tile 1x1", "black", 35 * m, analysis, "feature_details", "pupils and dark outlines")

    add_part_line(parts, "Tile 1x1", eye_color, 20 * m, analysis, "feature_details", "eye color detail")

    add_part_line(parts, "Tile 1x1", "white", 60 * m, analysis, "feature_details", "white muzzle, paws, and highlights")
    add_part_line(parts, "Plate 1x1", "red", 6 * m, analysis, "feature_details", "nose or warm facial detail")
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


def generate_vehicle_plan_from_analysis(analysis, data):
    parts = []
    m = get_tier_multiplier(analysis)

    add_part_line(parts, "Plate 2x2", "black", 20 * m, analysis, "chassis", "vehicle base")
    add_part_line(parts, "Brick 1x2", "light_bluish_gray", 80 * m, analysis, "body_shell", "main body shell")
    add_part_line(parts, "Brick 1x4", "light_bluish_gray", 40 * m, analysis, "body_shell", "long body sections")
    add_part_line(parts, "Tile 1x2", "black", 30 * m, analysis, "windows_lights", "windows and dark details")
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


def generate_generic_plan_from_analysis(analysis, data):
    parts = []
    m = get_tier_multiplier(analysis)

    add_part_line(parts, "Plate 2x2", "light_bluish_gray", 40 * m, analysis, "display_base", "simple display base")
    add_part_line(parts, "Brick 1x2", "light_bluish_gray", 80 * m, analysis, "main_subject", "main subject body")
    add_part_line(parts, "Plate 1x2", "dark_bluish_gray", 40 * m, analysis, "main_subject", "shadow and detail layers")
    add_part_line(parts, "Tile 1x2", "tan", 30 * m, analysis, "main_subject", "surface color detail")

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
        }
    ]

    return make_response(
        "Generic analysis-based brick plan generated",
        analysis,
        build_modules,
        parts,
        data
    )


def generate_from_analysis(data):
    analysis = data.get("analysis", {})

    model_type = analysis.get("recommended_model_type", "")
    category = analysis.get("category", "")

    if model_type == "mosaic_relief":
        return generate_mosaic_plan_from_analysis(analysis, data)

    if model_type in ["architecture_full_model", "architecture_facade"]:
        return generate_architecture_plan_from_analysis(analysis, data)

    if model_type == "vehicle_model":
        return generate_vehicle_plan_from_analysis(analysis, data)

    if category in ["pet_animal", "person_portrait"]:
        return generate_mosaic_plan_from_analysis(analysis, data)

    return generate_generic_plan_from_analysis(analysis, data)


@app.post("/generate-lego-model")
async def generate_lego_model(data: dict):
    analysis = data.get("analysis")

    if not analysis:
        return {
            "error": "Missing analysis JSON",
            "expected_body": {
                "analysis": {},
                "include_bricklink_parts": False,
                "include_wanted_list_xml": True,
                "include_build_modules": True,
                "detail_level": 2,
                "validate_with_rebrickable": True
            }
        }

    return generate_from_analysis(data)
