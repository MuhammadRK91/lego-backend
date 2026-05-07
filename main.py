from fastapi import FastAPI
import os
import requests

app = FastAPI()

REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3"


# Starter candidate library.
# These are common LEGO/Rebrickable/BrickLink-compatible part numbers.
# Rebrickable will validate whether they exist and whether the color exists for the part.
PARTS = {
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
}


# Rebrickable color IDs.
# Many of these also match common BrickLink color IDs for basic colors,
# but treat Rebrickable as the validation source.
COLORS = {
    "white": 1,
    "tan": 2,
    "red": 4,
    "green": 6,
    "blue": 7,
    "black": 0,
    "dark_green": 80,
    "dark_bluish_gray": 72,
    "light_bluish_gray": 71,
    "reddish_brown": 70,
    "dark_tan": 19,
    "yellow": 14,
    "orange": 25,
    "brown": 8,
    "light_gray": 9
}


@app.get("/")
def root():
    return {
        "status": "running",
        "mode": "analysis_only",
        "catalog_source": "rebrickable",
        "rebrickable_api_configured": rebrickable_is_configured()
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


def get_tier_multiplier(analysis):
    tier = str(analysis.get("recommended_pricing_tier", "")).lower()
    model_type = str(analysis.get("recommended_model_type", "")).lower()

    if tier == "premium":
        return 3

    if tier == "standard":
        return 2

    return 1


def add_part_line(parts, part_name, color, qty):
    part_num = PARTS.get(part_name)
    color_id = COLORS.get(color, 71)

    parts.append({
        "part_name": part_name,
        "part_num": part_num,
        "bricklink_id": part_num,
        "color": color,
        "rebrickable_color_id": color_id,
        "bricklink_color_id": color_id,
        "quantity": int(qty),
        "validation": {
            "checked": False,
            "part_exists": None,
            "color_exists_for_part": None,
            "official_part_name": None,
            "notes": []
        }
    })


def build_parts_summary(parts):
    summary = {}

    for p in parts:
        key = f"{p['part_name']} - {p['color']}"
        summary[key] = summary.get(key, 0) + int(p["quantity"])

    return summary


def create_bricklink_xml(parts):
    xml = "<INVENTORY>\n"

    for p in parts:
        if not p.get("bricklink_id"):
            continue

        xml += "  <ITEM>\n"
        xml += "    <ITEMTYPE>P</ITEMTYPE>\n"
        xml += f"    <ITEMID>{p['bricklink_id']}</ITEMID>\n"
        xml += f"    <COLOR>{p['bricklink_color_id']}</COLOR>\n"
        xml += f"    <MINQTY>{p['quantity']}</MINQTY>\n"
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
            p["validation"]["notes"].append("Missing part number.")
            continue

        try:
            part_data = rebrickable_get(f"/lego/parts/{part_num}/")

            p["validation"]["checked"] = True
            p["validation"]["part_exists"] = True
            p["validation"]["official_part_name"] = part_data.get("name")

        except Exception as e:
            p["validation"]["checked"] = True
            p["validation"]["part_exists"] = False
            p["validation"]["color_exists_for_part"] = False
            p["validation"]["notes"].append(f"Part lookup failed: {str(e)}")
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

    if validate:
        parts = validate_parts_with_rebrickable(parts)

    return {
        "message": message,
        "generation_mode": "analysis_only",
        "catalog_source": "rebrickable",
        "rebrickable_api_configured": rebrickable_is_configured(),
        "rebrickable_validation_enabled": validate,
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
        "bricklink_wanted_list_xml": create_bricklink_xml(parts),
        "source_analysis": analysis
    }


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
    add_part_line(parts, "Tile 1x1", "green", 20 * m)
    add_part_line(parts, "Tile 1x1", "white", 60 * m)
    add_part_line(parts, "Plate 1x1", "red", 6 * m)
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
                "include_bricklink_parts": True,
                "include_wanted_list_xml": True,
                "include_build_modules": True,
                "detail_level": 2,
                "validate_with_rebrickable": True
            }
        }

    return generate_from_analysis(data)
