from fastapi import FastAPI
import os
import re
import requests
from functools import lru_cache

app = FastAPI()

REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3"


@app.get("/")
def root():
    return {
        "status": "running",
        "mode": "analysis_only",
        "catalog_source": "rebrickable",
        "rebrickable_api_configured": rebrickable_is_configured(),
        "bricklink_api_used": False,
        "note": "Parts and colors are resolved through Rebrickable API, not hardcoded IDs."
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
        "light green": "green",
        "hazel": "green"
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
    Converts a simple planner color name like 'tan' or 'light_bluish_gray'
    into an actual Rebrickable color object from the API.
    """
    normalized = normalize_color_alias(color_name)
    color_map = get_rebrickable_colors_cache()["by_normalized_name"]

    if normalized in color_map:
        return color_map[normalized]

    # Safe fallback: Light Bluish Gray if available.
    fallback = color_map.get("light bluish gray")

    if fallback:
        return fallback

    # Final fallback if color catalog is somehow missing expected color.
    return {
        "id": None,
        "name": color_name,
        "rgb": None,
        "is_trans": False
    }


@lru_cache(maxsize=512)
def search_rebrickable_part(search_term):
    """
    Finds a part number from Rebrickable by search term.
    This avoids hardcoding part IDs like 3023, 3024, etc.
    """
    data = rebrickable_get(
        "/lego/parts/",
        params={
            "search": search_term,
            "page_size": 10
        }
    )

    results = data.get("results", [])

    if not results:
        return None

    normalized_search = normalize_text(search_term)

    # Prefer exact-ish name match.
    for item in results:
        item_name = normalize_text(item.get("name", ""))

        if normalized_search in item_name or item_name in normalized_search:
            return {
                "part_num": item.get("part_num"),
                "official_part_name": item.get("name"),
                "part_img_url": item.get("part_img_url")
            }

    # Otherwise use first result.
    first = results[0]

    return {
        "part_num": first.get("part_num"),
        "official_part_name": first.get("name"),
        "part_img_url": first.get("part_img_url")
    }


def resolve_part(part_name):
    """
    Converts internal planner labels into Rebrickable search terms.
    These are not hardcoded IDs. They are search labels used to ask Rebrickable.
    """
    search_terms = {
        "Plate 1x1": "Plate 1 x 1",
        "Plate 1x2": "Plate 1 x 2",
        "Plate 2x2": "Plate 2 x 2",
        "Tile 1x1": "Tile 1 x 1 with Groove",
        "Tile 1x2": "Tile 1 x 2 with Groove",
        "Tile 2x2": "Tile 2 x 2 with Groove",
        "Brick 1x1": "Brick 1 x 1",
        "Brick 1x2": "Brick 1 x 2",
        "Brick 1x4": "Brick 1 x 4",
        "Brick Round 2x2": "Brick Round 2 x 2",
        "Arch 1x4": "Arch 1 x 4"
    }

    search_term = search_terms.get(part_name, part_name)

    try:
        return search_rebrickable_part(search_term)
    except Exception as e:
        return {
            "part_num": None,
            "official_part_name": None,
            "part_img_url": None,
            "error": str(e)
        }


def get_tier_multiplier(analysis):
    tier = str(analysis.get("recommended_pricing_tier", "")).lower()

    if tier == "premium":
        return 3

    if tier == "standard":
        return 2

    return 1


def add_part_line(parts, part_name, color, qty):
    color_data = resolve_rebrickable_color(color)
    part_data = resolve_part(part_name)

    part_num = None
    official_part_name = None
    part_img_url = None
    resolve_error = None

    if part_data:
        part_num = part_data.get("part_num")
        official_part_name = part_data.get("official_part_name")
        part_img_url = part_data.get("part_img_url")
        resolve_error = part_data.get("error")

    notes = []

    if resolve_error:
        notes.append(f"Part search failed: {resolve_error}")

    if not part_num:
        notes.append(f"Could not resolve part from Rebrickable search for planner part: {part_name}")

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
        "validation": {
            "checked": False,
            "part_exists": None,
            "color_exists_for_part": None,
            "official_part_name": official_part_name,
            "notes": notes
        },
        "part_img_url": part_img_url
    })


def build_parts_summary(parts):
    summary = {}

    for p in parts:
        color_name = p.get("rebrickable_color_name") or p.get("color")
        key = f"{p['part_name']} - {color_name}"
        summary[key] = summary.get(key, 0) + int(p["quantity"])

    return summary


def create_rebrickable_parts_export(parts):
    """
    Simple export that is based on Rebrickable part_num + Rebrickable color_id.
    This is safer than pretending we are producing BrickLink-validated XML.
    """
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
    Generic XML export using Rebrickable-resolved part numbers and color IDs.
    This is not labeled as BrickLink XML because BrickLink color IDs may differ.
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

            if part_data.get("part_img_url") and not p.get("part_img_url"):
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
        "bricklink_api_used": False,
        "bricklink_xml_export_available": False,
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

    add_part_line(parts, "Plate 2x2", "green", 40 * m)
    add_part_line(parts, "Plate 1x2", "dark_green", 30 * m)
    add_part_line(parts, "Plate 1x1", "red", 16 * m)

    add_part_line(parts, "Brick Round 2x2", "dark_bluish_gray", 90 * m)
    add_part_line(parts, "Brick 1x2", "dark_bluish_gray", 180 * m)
    add_part_line(parts, "Brick 1x4", "dark_bluish_gray", 80 * m)
    add_part_line(parts, "Plate 1x1", "black", 28 * m)

    add_part_line(parts, "Arch 1x4", "tan", 8 * m)
    add_part_line(parts, "Plate 1x2", "tan", 60 * m)
    add_part_line(parts, "Tile 1x2", "white", 30 * m)

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

    add_part_line(parts, "Tile 2x2", "tan", 80 * m)
    add_part_line(parts, "Tile 1x2", "reddish_brown", 90 * m)
    add_part_line(parts, "Tile 1x1", "black", 35 * m)

    # Eye color for pets/portraits.
    add_part_line(parts, "Tile 1x1", "green", 20 * m)

    add_part_line(parts, "Tile 1x1", "white", 60 * m)

    # Nose / small warm detail.
    add_part_line(parts, "Plate 1x1", "red", 6 * m)

    # Neutral background/base grid.
    add_part_line(parts, "Plate 1x2", "light_bluish_gray", 80 * m)

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

    add_part_line(parts, "Plate 2x2", "black", 20 * m)
    add_part_line(parts, "Brick 1x2", "light_bluish_gray", 80 * m)
    add_part_line(parts, "Brick 1x4", "light_bluish_gray", 40 * m)
    add_part_line(parts, "Tile 1x2", "black", 30 * m)
    add_part_line(parts, "Plate 1x1", "red", 8 * m)

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

    add_part_line(parts, "Plate 2x2", "light_bluish_gray", 40 * m)
    add_part_line(parts, "Brick 1x2", "light_bluish_gray", 80 * m)
    add_part_line(parts, "Plate 1x2", "dark_bluish_gray", 40 * m)
    add_part_line(parts, "Tile 1x2", "tan", 30 * m)

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
