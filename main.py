from fastapi import FastAPI

app = FastAPI()


BRICKLINK_PARTS = {
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

BRICKLINK_COLORS = {
    "white": "1",
    "tan": "2",
    "red": "5",
    "green": "6",
    "blue": "7",
    "black": "11",
    "dark_green": "80",
    "dark_bluish_gray": "85",
    "light_bluish_gray": "86",
    "reddish_brown": "88",
    "dark_tan": "69"
}


@app.get("/")
def root():
    return {"status": "running", "mode": "analysis_only"}


def add_part_line(parts, part_name, color, qty):
    parts.append({
        "part_name": part_name,
        "bricklink_id": BRICKLINK_PARTS.get(part_name),
        "color": color,
        "bricklink_color_id": BRICKLINK_COLORS.get(color, "86"),
        "quantity": int(qty)
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


def make_response(message, analysis, build_modules, parts):
    return {
        "message": message,
        "generation_mode": "analysis_only",
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


def generate_architecture_plan_from_analysis(analysis):
    parts = []

    add_part_line(parts, "Plate 2x2", "green", 40)
    add_part_line(parts, "Plate 1x2", "dark_green", 30)
    add_part_line(parts, "Plate 1x1", "red", 16)

    add_part_line(parts, "Brick Round 2x2", "dark_bluish_gray", 90)
    add_part_line(parts, "Brick 1x2", "dark_bluish_gray", 180)
    add_part_line(parts, "Brick 1x4", "dark_bluish_gray", 80)
    add_part_line(parts, "Plate 1x1", "black", 28)

    add_part_line(parts, "Arch 1x4", "tan", 8)
    add_part_line(parts, "Plate 1x2", "tan", 60)
    add_part_line(parts, "Tile 1x2", "white", 30)

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
        parts
    )


def generate_mosaic_plan_from_analysis(analysis):
    parts = []

    add_part_line(parts, "Tile 2x2", "tan", 80)
    add_part_line(parts, "Tile 1x2", "reddish_brown", 90)
    add_part_line(parts, "Tile 1x1", "black", 35)
    add_part_line(parts, "Tile 1x1", "green", 20)
    add_part_line(parts, "Tile 1x1", "white", 60)
    add_part_line(parts, "Plate 1x1", "red", 6)
    add_part_line(parts, "Plate 1x2", "light_bluish_gray", 80)

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
        parts
    )


def generate_vehicle_plan_from_analysis(analysis):
    parts = []

    add_part_line(parts, "Plate 2x2", "black", 20)
    add_part_line(parts, "Brick 1x2", "light_bluish_gray", 80)
    add_part_line(parts, "Brick 1x4", "light_bluish_gray", 40)
    add_part_line(parts, "Tile 1x2", "black", 30)
    add_part_line(parts, "Plate 1x1", "red", 8)

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
        parts
    )


def generate_generic_plan_from_analysis(analysis):
    parts = []

    add_part_line(parts, "Plate 2x2", "light_bluish_gray", 40)
    add_part_line(parts, "Brick 1x2", "light_bluish_gray", 80)
    add_part_line(parts, "Plate 1x2", "dark_bluish_gray", 40)
    add_part_line(parts, "Tile 1x2", "tan", 30)

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
        parts
    )


def generate_from_analysis(data):
    analysis = data.get("analysis", {})

    model_type = analysis.get("recommended_model_type", "")
    category = analysis.get("category", "")

    if model_type == "mosaic_relief":
        return generate_mosaic_plan_from_analysis(analysis)

    if model_type in ["architecture_full_model", "architecture_facade"]:
        return generate_architecture_plan_from_analysis(analysis)

    if model_type == "vehicle_model":
        return generate_vehicle_plan_from_analysis(analysis)

    if category in ["pet_animal", "person_portrait"]:
        return generate_mosaic_plan_from_analysis(analysis)

    return generate_generic_plan_from_analysis(analysis)


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
                "detail_level": 2
            }
        }

    return generate_from_analysis(data)
