from fastapi import FastAPI
import requests
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

GRID_WIDTH = 48
GRID_HEIGHT = 32
MAX_HEIGHT_PLATES = 12

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/generate-lego-model")
async def generate_lego_model(data: dict):
    depth_url = data.get("depth_map_url")

    depth_img = Image.open(BytesIO(requests.get(depth_url).content)).convert("L")
    depth_img = depth_img.resize((GRID_WIDTH, GRID_HEIGHT))

    arr = np.array(depth_img)
    height_map = np.round((arr / 255) * MAX_HEIGHT_PLATES).astype(int)

    brick_layout = []

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            height_plates = int(height_map[y][x])

            if height_plates > 0:
                full_bricks = height_plates // 3
                extra_plates = height_plates % 3

                brick_layout.append({
                    "x": x,
                    "y": y,
                    "height_plates": height_plates,
                    "full_bricks_1x1": full_bricks,
                    "extra_plates_1x1": extra_plates,
                    "color": "light_bluish_gray"
                })

    return {
        "message": "LEGO brick layout generated",
        "grid_size": {
            "width": GRID_WIDTH,
            "height": GRID_HEIGHT
        },
        "max_height_plates": MAX_HEIGHT_PLATES,
        "total_positions": GRID_WIDTH * GRID_HEIGHT,
        "used_positions": len(brick_layout),
        "height_map": height_map.tolist(),
        "brick_layout": brick_layout
    }
