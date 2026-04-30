from fastapi import FastAPI
import requests
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO
import base64

app = FastAPI()

GRID_WIDTH = 96
GRID_HEIGHT = 64
MAX_HEIGHT_PLATES = 12

LEGO_COLORS = {
    "white": (242, 243, 242),
    "light_bluish_gray": (160, 165, 169),
    "dark_bluish_gray": (99, 95, 98),
    "black": (27, 42, 52),
    "dark_green": (0, 69, 26),
    "green": (35, 120, 65),
    "tan": (215, 197, 153),
    "reddish_brown": (88, 42, 18),
    "dark_blue": (0, 32, 96),
}

def closest_lego_color(rgb):
    r, g, b = rgb
    best_name = "light_bluish_gray"
    best_distance = float("inf")

    for name, color in LEGO_COLORS.items():
        cr, cg, cb = color
        distance = (int(r) - cr) ** 2 + (int(g) - cg) ** 2 + (int(b) - cb) ** 2

        if distance < best_distance:
            best_distance = distance
            best_name = name

    return best_name

def add_part(parts_summary, part_name, color, quantity=1):
    key = f"{part_name} - {color}"
    parts_summary[key] = parts_summary.get(key, 0) + quantity

def create_preview_image(brick_layout):
    cell_size = 8
    img = Image.new("RGB", (GRID_WIDTH * cell_size, GRID_HEIGHT * cell_size), "white")
    draw = ImageDraw.Draw(img)

    for brick in brick_layout:
        x = brick["x"]
        y = brick["y"]
        color = brick["color"]
        height = brick["height_plates"]

        rgb = LEGO_COLORS.get(color, LEGO_COLORS["light_bluish_gray"])

        # small height effect: taller bricks appear slightly darker
        shade = max(0.55, 1 - (height / MAX_HEIGHT_PLATES) * 0.35)
        shaded_rgb = tuple(int(c * shade) for c in rgb)

        x1 = x * cell_size
        y1 = y * cell_size
        x2 = x1 + cell_size - 1
        y2 = y1 + cell_size - 1

        draw.rectangle([x1, y1, x2, y2], fill=shaded_rgb)
        draw.rectangle([x1, y1, x2, y2], outline=(230, 230, 230))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def optimize_row(row_cells, part_type, parts_summary):
    optimized = []
    x = 0

    while x < len(row_cells):
        cell = row_cells[x]

        if cell is None:
            x += 1
            continue

        color = cell["color"]
        quantity = cell["quantity"]
        start_x = x

        run_length = 1

        while (
            x + run_length < len(row_cells)
            and row_cells[x + run_length] is not None
            and row_cells[x + run_length]["color"] == color
            and row_cells[x + run_length]["quantity"] == quantity
        ):
            run_length += 1

        remaining = run_length
        current_x = start_x

        for size in [4, 3, 2, 1]:
            while remaining >= size:
                part_name = f"{part_type} 1x{size}"
                add_part(parts_summary, part_name, color, quantity)

                optimized.append({
                    "x": current_x,
                    "y": cell["y"],
                    "part": part_name,
                    "color": color,
                    "quantity": quantity
                })

                current_x += size
                remaining -= size

        x = start_x + run_length

    return optimized

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/generate-lego-model")
async def generate_lego_model(data: dict):
    depth_url = data.get("depth_map_url")
    original_url = data.get("original_image_url")

    if not depth_url:
        return {"error": "Missing depth_map_url"}

    if not original_url:
        return {"error": "Missing original_image_url"}

    depth_response = requests.get(depth_url)
    original_response = requests.get(original_url)

    depth_img = Image.open(BytesIO(depth_response.content)).convert("L")
    depth_img = depth_img.resize((GRID_WIDTH, GRID_HEIGHT))

    original_img = Image.open(BytesIO(original_response.content)).convert("RGB")
    original_img = original_img.resize((GRID_WIDTH, GRID_HEIGHT))

    depth_arr = np.array(depth_img)
    color_arr = np.array(original_img)

    height_map = np.round((depth_arr / 255) * MAX_HEIGHT_PLATES).astype(int)

    brick_layout = []
    optimized_parts = []
    parts_summary = {}

    for y in range(GRID_HEIGHT):
        brick_row = [None] * GRID_WIDTH
        plate_row = [None] * GRID_WIDTH

        for x in range(GRID_WIDTH):
            height_plates = int(height_map[y][x])

            if height_plates > 0:
                rgb = tuple(color_arr[y][x])
                color = closest_lego_color(rgb)

                full_bricks = height_plates // 3
                extra_plates = height_plates % 3

                brick_layout.append({
                    "x": x,
                    "y": y,
                    "height_plates": height_plates,
                    "full_bricks": full_bricks,
                    "extra_plates": extra_plates,
                    "color": color
                })

                if full_bricks > 0:
                    brick_row[x] = {
                        "x": x,
                        "y": y,
                        "color": color,
                        "quantity": full_bricks
                    }

                if extra_plates > 0:
                    plate_row[x] = {
                        "x": x,
                        "y": y,
                        "color": color,
                        "quantity": extra_plates
                    }

        optimized_parts.extend(optimize_row(brick_row, "Brick", parts_summary))
        optimized_parts.extend(optimize_row(plate_row, "Plate", parts_summary))

    preview_image_base64 = create_preview_image(brick_layout)

    return {
        "message": "LEGO optimized colored brick layout generated",
        "grid_size": {
            "width": GRID_WIDTH,
            "height": GRID_HEIGHT
        },
        "max_height_plates": MAX_HEIGHT_PLATES,
        "used_positions": len(brick_layout),
        "parts_summary": parts_summary,
        "preview_image_base64": preview_image_base64,
        "brick_layout": brick_layout,
        "optimized_parts": optimized_parts
    }
