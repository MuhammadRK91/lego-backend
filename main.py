from fastapi import FastAPI
import requests
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

GRID_WIDTH = 48
GRID_HEIGHT = 32
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
    "dark_blue": (0, 32, 96)
}

def closest_lego_color(rgb):
    r, g, b = rgb
    best_name = "light_bluish_gray"
    best_distance = 999999

    for name, color in LEGO_COLORS.items():
        cr, cg, cb = color
        distance = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2

        if distance < best_distance:
            best_distance = distance
            best_name = name

    return best_name

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/generate-lego-model")
async def generate_lego_model(data: dict):
    depth_url = data.get("depth_map_url")
    original_url = data.get("original_image_url")

    depth_img = Image.open(BytesIO(requests.get(depth_url).content)).convert("L")
    depth_img = depth_img.resize((GRID_WIDTH, GRID_HEIGHT))

    original_img = Image.open(BytesIO(requests.get(original_url).content)).convert("RGB")
    original_img = original_img.resize((GRID_WIDTH, GRID_HEIGHT))

    depth_arr = np.array(depth_img)
    color_arr = np.array(original_img)

    height_map = np.round((depth_arr / 255) * MAX_HEIGHT_PLATES).astype(int)

    brick_layout = []
    parts_summary = {}

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            height_plates = int(height_map[y][x])

            if height_plates > 0:
                rgb = tuple(color_arr[y][x])
                color = closest_lego_color(rgb)

                full_bricks = height_plates // 3
                extra_plates = height_plates % 3

                if full_bricks > 0:
                    key = f"Brick 1x1 - {color}"
                    parts_summary[key] = parts_summary.get(key, 0) + full_bricks

                if extra_plates > 0:
                    key = f"Plate 1x1 - {color}"
                    parts_summary[key] = parts_summary.get(key, 0) + extra_plates

                brick_layout.append({
                    "x": x,
                    "y": y,
                    "height_plates": height_plates,
                    "full_bricks_1x1": full_bricks,
                    "extra_plates_1x1": extra_plates,
                    "color": color
                })

    return {
        "message": "LEGO colored brick layout generated",
        "grid_size": {
            "width": GRID_WIDTH,
            "height": GRID_HEIGHT
        },
        "max_height_plates": MAX_HEIGHT_PLATES,
        "used_positions": len(brick_layout),
        "parts_summary": parts_summary,
        "brick_layout": brick_layout
    }
