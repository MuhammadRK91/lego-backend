from fastapi import FastAPI
import requests
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from io import BytesIO
import base64
import math

app = FastAPI()

GRID_WIDTH = 96
GRID_HEIGHT = 64
MAX_HEIGHT_PLATES = 12
CELL_SIZE = 8

LEGO_COLORS = {
    "white": (242, 243, 242),
    "light_bluish_gray": (160, 165, 169),
    "dark_bluish_gray": (99, 95, 98),
    "black": (27, 42, 52),
    "tan": (215, 197, 153),
    "dark_tan": (149, 138, 115),
    "reddish_brown": (88, 42, 18),
    "brown": (124, 80, 58),
    "dark_red": (123, 46, 47),
    "red": (196, 40, 28),
    "dark_blue": (0, 32, 96),
    "blue": (13, 105, 171),
    "sand_blue": (96, 116, 161),
    "dark_green": (0, 69, 26),
    "green": (35, 120, 65),
    "olive_green": (155, 154, 90),
    "lime": (187, 233, 11),
    "yellow": (245, 205, 47),
    "orange": (218, 133, 64),
}

def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def closest_lego_color(rgb):
    r, g, b = rgb
    best_name = "light_bluish_gray"
    best_distance = float("inf")

    for name, color in LEGO_COLORS.items():
        cr, cg, cb = color
        distance = (
            (int(r) - cr) ** 2 +
            (int(g) - cg) ** 2 +
            (int(b) - cb) ** 2
        )

        if distance < best_distance:
            best_distance = distance
            best_name = name

    return best_name

def add_part(parts_summary, part_name, color, quantity=1):
    key = f"{part_name} - {color}"
    parts_summary[key] = parts_summary.get(key, 0) + quantity

def smooth_height_map(height_map):
    padded = np.pad(height_map, 1, mode="edge")
    smoothed = np.zeros_like(height_map)

    for y in range(height_map.shape[0]):
        for x in range(height_map.shape[1]):
            region = padded[y:y+3, x:x+3]
            smoothed[y, x] = round(np.mean(region))

    return smoothed.astype(int)

def boost_edges(depth_arr, height_map):
    gy, gx = np.gradient(depth_arr.astype(float))
    edges = np.sqrt(gx * gx + gy * gy)
    edges = edges / max(edges.max(), 1)

    boosted = height_map.copy()
    boosted[edges > 0.18] += 1
    boosted = np.clip(boosted, 0, MAX_HEIGHT_PLATES)

    return boosted.astype(int)

def create_top_preview(brick_layout):
    img = Image.new(
        "RGB",
        (GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE),
        "white"
    )
    draw = ImageDraw.Draw(img)

    for brick in brick_layout:
        x = brick["x"]
        y = brick["y"]
        color = brick["color"]
        height = brick["height_plates"]

        rgb = LEGO_COLORS.get(color, LEGO_COLORS["light_bluish_gray"])

        shade = max(0.55, 1 - (height / MAX_HEIGHT_PLATES) * 0.30)
        shaded_rgb = tuple(int(c * shade) for c in rgb)

        x1 = x * CELL_SIZE
        y1 = y * CELL_SIZE
        x2 = x1 + CELL_SIZE - 1
        y2 = y1 + CELL_SIZE - 1

        draw.rectangle([x1, y1, x2, y2], fill=shaded_rgb)
        draw.rectangle([x1, y1, x2, y2], outline=(225, 225, 225))

    return image_to_base64(img)

def create_isometric_preview(brick_layout):
    tile_w = 10
    tile_h = 6
    height_step = 2

    canvas_w = GRID_WIDTH * tile_w + GRID_HEIGHT * tile_w + 200
    canvas_h = GRID_HEIGHT * tile_h + GRID_WIDTH * tile_h + MAX_HEIGHT_PLATES * height_step + 200

    img = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    offset_x = canvas_w // 2
    offset_y = 80

    sorted_bricks = sorted(brick_layout, key=lambda b: b["x"] + b["y"])

    for brick in sorted_bricks:
        x = brick["x"]
        y = brick["y"]
        h = brick["height_plates"]
        color = brick["color"]

        rgb = LEGO_COLORS.get(color, LEGO_COLORS["light_bluish_gray"])

        iso_x = offset_x + (x - y) * tile_w // 2
        iso_y = offset_y + (x + y) * tile_h // 2 - h * height_step

        top = [
            (iso_x, iso_y),
            (iso_x + tile_w // 2, iso_y + tile_h // 2),
            (iso_x, iso_y + tile_h),
            (iso_x - tile_w // 2, iso_y + tile_h // 2),
        ]

        left = [
            (iso_x - tile_w // 2, iso_y + tile_h // 2),
            (iso_x, iso_y + tile_h),
            (iso_x, iso_y + tile_h + h * height_step),
            (iso_x - tile_w // 2, iso_y + tile_h // 2 + h * height_step),
        ]

        right = [
            (iso_x + tile_w // 2, iso_y + tile_h // 2),
            (iso_x, iso_y + tile_h),
            (iso_x, iso_y + tile_h + h * height_step),
            (iso_x + tile_w // 2, iso_y + tile_h // 2 + h * height_step),
        ]

        left_rgb = tuple(max(0, int(c * 0.65)) for c in rgb)
        right_rgb = tuple(max(0, int(c * 0.80)) for c in rgb)
        top_rgb = rgb

        draw.polygon(left, fill=left_rgb)
        draw.polygon(right, fill=right_rgb)
        draw.polygon(top, fill=top_rgb)

        draw.line(top + [top[0]], fill=(60, 60, 60), width=1)

        stud_r = 2
        draw.ellipse(
            [
                iso_x - stud_r,
                iso_y + tile_h // 2 - stud_r,
                iso_x + stud_r,
                iso_y + tile_h // 2 + stud_r,
            ],
            fill=tuple(min(255, int(c * 1.15)) for c in rgb),
            outline=(80, 80, 80),
        )

    return image_to_base64(img)

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

        for size in [8, 6, 4, 3, 2, 1]:
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
    depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=0.6))

    original_img = Image.open(BytesIO(original_response.content)).convert("RGB")
    original_img = original_img.resize((GRID_WIDTH, GRID_HEIGHT))

    depth_arr = np.array(depth_img)
    color_arr = np.array(original_img)

    height_map = np.round((depth_arr / 255) * MAX_HEIGHT_PLATES).astype(int)
    height_map = smooth_height_map(height_map)
    height_map = boost_edges(depth_arr, height_map)

    brick_layout = []
    optimized_parts = []
    parts_summary = {}
    color_summary = {}
    height_summary = {}

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

                color_summary[color] = color_summary.get(color, 0) + 1
                height_summary[str(height_plates)] = height_summary.get(str(height_plates), 0) + 1

                brick_layout.append({
                    "x": x,
                    "y": y,
                    "height_plates": height_plates,
                    "full_bricks": full_bricks,
                    "extra_plates": extra_plates,
                    "color": color,
                    "rgb": LEGO_COLORS[color]
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

    preview_image_base64 = create_top_preview(brick_layout)
    isometric_preview_base64 = create_isometric_preview(brick_layout)

    total_parts = sum(parts_summary.values())
    total_positions = GRID_WIDTH * GRID_HEIGHT
    used_positions = len(brick_layout)

    model_stats = {
        "grid_width": GRID_WIDTH,
        "grid_height": GRID_HEIGHT,
        "total_positions": total_positions,
        "used_positions": used_positions,
        "empty_positions": total_positions - used_positions,
        "coverage_percent": round((used_positions / total_positions) * 100, 2),
        "max_height_plates": MAX_HEIGHT_PLATES,
        "estimated_total_parts": total_parts,
        "unique_part_types": len(parts_summary),
        "unique_colors": len(color_summary)
    }

    return {
        "message": "Advanced LEGO model generated",
        "model_stats": model_stats,
        "grid_size": {
            "width": GRID_WIDTH,
            "height": GRID_HEIGHT
        },
        "max_height_plates": MAX_HEIGHT_PLATES,
        "parts_summary": parts_summary,
        "color_summary": color_summary,
        "height_summary": height_summary,
        "preview_image_base64": preview_image_base64,
        "isometric_preview_base64": isometric_preview_base64,
        "brick_layout": brick_layout,
        "optimized_parts": optimized_parts
    }
