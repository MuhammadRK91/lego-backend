from fastapi import FastAPI
import requests
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from io import BytesIO
import base64
import cv2

app = FastAPI()

GRID_WIDTH = 256
GRID_HEIGHT = 192
MAX_HEIGHT_PLATES = 12
CELL_SIZE = 3

# Depth Anything V2 works better with a lower threshold than old MiDaS.
MIN_DEPTH_VALUE = 8

# Cleaner settings.
EDGE_DETAIL_BOOST = 1
DEPTH_EDGE_BOOST = 1
WINDOW_RECESS_AMOUNT = 1
BRIGHT_DETAIL_BOOST = 1

LEGO_COLORS = {
    "white": (242, 243, 242),
    "very_light_gray": (230, 230, 225),
    "light_bluish_gray": (160, 165, 169),
    "medium_gray": (130, 136, 140),
    "dark_bluish_gray": (99, 95, 98),
    "black": (27, 42, 52),

    "tan": (215, 197, 153),
    "light_tan": (240, 220, 180),
    "dark_tan": (149, 138, 115),
    "sand_yellow": (160, 140, 90),

    "reddish_brown": (88, 42, 18),
    "brown": (124, 80, 58),
    "dark_brown": (60, 30, 10),

    "dark_red": (123, 46, 47),
    "red": (196, 40, 28),
    "dark_orange": (160, 95, 40),
    "orange": (218, 133, 64),

    "dark_blue": (0, 32, 96),
    "blue": (13, 105, 171),
    "medium_blue": (90, 147, 219),
    "sand_blue": (96, 116, 161),

    "dark_green": (0, 69, 26),
    "green": (35, 120, 65),
    "sand_green": (120, 144, 130),
    "olive_green": (155, 154, 90),
    "lime": (187, 233, 11),

    "yellow": (245, 205, 47),
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
        distance = (int(r) - cr) ** 2 + (int(g) - cg) ** 2 + (int(b) - cb) ** 2

        if distance < best_distance:
            best_distance = distance
            best_name = name

    return best_name


def add_part(parts_summary, part_name, color, quantity=1):
    key = f"{part_name} - {color}"
    parts_summary[key] = parts_summary.get(key, 0) + quantity


def lighten(rgb, factor=1.15):
    return tuple(min(255, int(c * factor)) for c in rgb)


def darken(rgb, factor=0.75):
    return tuple(max(0, int(c * factor)) for c in rgb)


def normalize_depth(depth_img):
    depth_arr = np.array(depth_img).astype(np.float32)

    p2 = np.percentile(depth_arr, 2)
    p98 = np.percentile(depth_arr, 98)

    depth_arr = np.clip(depth_arr, p2, p98)
    depth_arr = (depth_arr - p2) / (p98 - p2 + 1e-6)

    # Improves mid-depth structure.
    depth_arr = depth_arr ** 0.75

    depth_arr = np.clip(depth_arr * 255, 0, 255).astype(np.uint8)
    return depth_arr


def create_edge_map(original_img):
    img = original_img.resize((GRID_WIDTH, GRID_HEIGHT), Image.Resampling.LANCZOS)
    arr = np.array(img)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Higher thresholds = cleaner, less noisy edge map.
    edges = cv2.Canny(gray, 80, 180)

    kernel_open = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open)

    kernel_dilate = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel_dilate, iterations=1)

    return edges.astype(np.float32) / 255.0


def create_sky_mask(original_img):
    img = original_img.resize((GRID_WIDTH, GRID_HEIGHT), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.uint8)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    blue_sky_mask = (b > 120) & (g > 95) & (r < 190) & ((b - r) > 20)
    white_bg_mask = (r > 225) & (g > 225) & (b > 225)

    sky_mask = blue_sky_mask | white_bg_mask

    sky_mask_uint8 = sky_mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)

    sky_mask_uint8 = cv2.morphologyEx(sky_mask_uint8, cv2.MORPH_OPEN, kernel)
    sky_mask_uint8 = cv2.morphologyEx(sky_mask_uint8, cv2.MORPH_CLOSE, kernel)

    return sky_mask_uint8 > 0


def create_dark_recess_mask(original_img):
    img = original_img.resize((GRID_WIDTH, GRID_HEIGHT), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.uint8)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Stricter threshold to avoid turning too many shadows into holes.
    dark_mask = gray < 45

    dark_mask_uint8 = dark_mask.astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)

    dark_mask_uint8 = cv2.morphologyEx(dark_mask_uint8, cv2.MORPH_OPEN, kernel)

    return dark_mask_uint8 > 0


def create_bright_detail_mask(original_img):
    img = original_img.resize((GRID_WIDTH, GRID_HEIGHT), Image.Resampling.LANCZOS)
    arr = np.array(img).astype(np.uint8)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    r = arr[:, :, 0]
    b = arr[:, :, 2]

    bright_mask = (gray > 185) & ~((b > 130) & (b > r + 20))

    bright_mask_uint8 = bright_mask.astype(np.uint8) * 255
    kernel = np.ones((2, 2), np.uint8)

    bright_mask_uint8 = cv2.morphologyEx(bright_mask_uint8, cv2.MORPH_OPEN, kernel)

    return bright_mask_uint8 > 0


def smooth_height_map(height_map):
    padded = np.pad(height_map, 1, mode="edge")
    smoothed = np.zeros_like(height_map)

    for y in range(height_map.shape[0]):
        for x in range(height_map.shape[1]):
            region = padded[y:y + 3, x:x + 3]
            smoothed[y, x] = round(np.mean(region))

    return smoothed.astype(int)


def boost_depth_edges(depth_arr, height_map):
    gy, gx = np.gradient(depth_arr.astype(float))
    edges = np.sqrt(gx * gx + gy * gy)
    edges = edges / max(edges.max(), 1)

    boosted = height_map.copy()
    boosted[edges > 0.10] += DEPTH_EDGE_BOOST
    boosted = np.clip(boosted, 0, MAX_HEIGHT_PLATES)

    return boosted.astype(int)


def build_improved_height_map(depth_img, original_img):
    depth_arr = normalize_depth(depth_img)

    edge_map = create_edge_map(original_img)
    sky_mask = create_sky_mask(original_img)
    dark_recess_mask = create_dark_recess_mask(original_img)
    bright_detail_mask = create_bright_detail_mask(original_img)

    height_map = np.round((depth_arr / 255.0) * MAX_HEIGHT_PLATES).astype(int)

    height_map[depth_arr < MIN_DEPTH_VALUE] = 0
    height_map = smooth_height_map(height_map)
    height_map = boost_depth_edges(depth_arr, height_map)

    # Fine architectural detail.
    height_map = height_map + np.round(edge_map * EDGE_DETAIL_BOOST).astype(int)

    # Recess dark windows/doors.
    height_map[dark_recess_mask] = np.maximum(
        height_map[dark_recess_mask] - WINDOW_RECESS_AMOUNT,
        0
    )

    # Raise bright arch/decorative stonework.
    height_map[bright_detail_mask] += BRIGHT_DETAIL_BOOST

    # Remove sky/background.
    height_map[sky_mask] = 0

    # Remove weak depth areas again.
    height_map[depth_arr < MIN_DEPTH_VALUE] = 0

    height_map = np.clip(height_map, 0, MAX_HEIGHT_PLATES).astype(int)

    return height_map, depth_arr, edge_map, sky_mask, dark_recess_mask, bright_detail_mask


def add_depth_shading(rgb, height_plates):
    shade = max(0.72, 1 - (height_plates / MAX_HEIGHT_PLATES) * 0.18)
    return tuple(int(c * shade) for c in rgb)


def create_reference_like_preview(brick_layout):
    img = Image.new("RGB", (GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE), "white")
    draw = ImageDraw.Draw(img)

    for brick in brick_layout:
        x = brick["x"]
        y = brick["y"]
        color = brick["color"]
        height = brick["height_plates"]

        rgb = LEGO_COLORS.get(color, LEGO_COLORS["light_bluish_gray"])
        shaded_rgb = add_depth_shading(rgb, height)

        x1 = x * CELL_SIZE
        y1 = y * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE

        draw.rectangle([x1, y1, x2, y2], fill=shaded_rgb)

    img = img.resize((GRID_WIDTH * 4, GRID_HEIGHT * 4), Image.Resampling.BICUBIC)
    img = img.filter(ImageFilter.SHARPEN)

    return image_to_base64(img)


def create_stud_preview(brick_layout):
    cell_size = 6
    img = Image.new("RGB", (GRID_WIDTH * cell_size, GRID_HEIGHT * cell_size), "white")
    draw = ImageDraw.Draw(img)

    for brick in brick_layout:
        x = brick["x"]
        y = brick["y"]
        color = brick["color"]
        height = brick["height_plates"]

        rgb = LEGO_COLORS.get(color, LEGO_COLORS["light_bluish_gray"])
        shaded_rgb = add_depth_shading(rgb, height)

        x1 = x * cell_size
        y1 = y * cell_size
        x2 = x1 + cell_size - 1
        y2 = y1 + cell_size - 1

        draw.rectangle([x1, y1, x2, y2], fill=shaded_rgb)
        draw.rectangle([x1, y1, x2, y2], outline=(215, 215, 215))

        stud_radius = 1
        cx = x1 + cell_size // 2
        cy = y1 + cell_size // 2
        stud_rgb = tuple(min(255, int(c * 1.12)) for c in shaded_rgb)

        draw.ellipse(
            [cx - stud_radius, cy - stud_radius, cx + stud_radius, cy + stud_radius],
            fill=stud_rgb,
            outline=(110, 110, 110)
        )

    return image_to_base64(img)


def draw_iso_brick(draw, sx, sy, height, rgb):
    w = 4
    d = 3
    h = max(1, int(height * 1.2))

    top = [
        (sx, sy - h),
        (sx + w, sy - h - d),
        (sx + 2 * w, sy - h),
        (sx + w, sy - h + d),
    ]

    left = [
        (sx, sy - h),
        (sx + w, sy - h + d),
        (sx + w, sy + d),
        (sx, sy),
    ]

    right = [
        (sx + w, sy - h + d),
        (sx + 2 * w, sy - h),
        (sx + 2 * w, sy),
        (sx + w, sy + d),
    ]

    top_color = lighten(rgb, 1.18)
    left_color = darken(rgb, 0.78)
    right_color = darken(rgb, 0.62)

    draw.polygon(left, fill=left_color)
    draw.polygon(right, fill=right_color)
    draw.polygon(top, fill=top_color)

    cx = sx + w
    cy = sy - h
    stud_color = lighten(rgb, 1.28)
    draw.ellipse(
        [cx - 1, cy - 1, cx + 1, cy + 1],
        fill=stud_color
    )


def create_isometric_preview(brick_layout):
    scale_x = 4
    scale_y = 3

    img_width = GRID_WIDTH * scale_x + 80
    img_height = GRID_HEIGHT * scale_y + 120

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    sorted_layout = sorted(brick_layout, key=lambda b: (b["y"], b["x"]))

    for brick in sorted_layout:
        x = brick["x"]
        y = brick["y"]
        color = brick["color"]
        height = brick["height_plates"]

        rgb = LEGO_COLORS.get(color, LEGO_COLORS["light_bluish_gray"])

        sx = 40 + x * scale_x
        sy = 60 + y * scale_y

        draw_iso_brick(draw, sx, sy, height, rgb)

    img = img.filter(ImageFilter.SHARPEN)

    return image_to_base64(img)


def create_debug_map_base64(arr):
    if arr.dtype == bool:
        img_arr = arr.astype(np.uint8) * 255
    else:
        arr = np.array(arr)
        if arr.max() <= 1:
            img_arr = (arr * 255).astype(np.uint8)
        else:
            img_arr = arr.astype(np.uint8)

    img = Image.fromarray(img_arr).convert("L")
    img = img.resize((GRID_WIDTH * 4, GRID_HEIGHT * 4), Image.Resampling.NEAREST)
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


def clamp_blender_detail_level(value):
    try:
        level = int(value)
    except Exception:
        level = 2

    if level < 1:
        level = 1

    if level > 4:
        level = 4

    return level


def get_most_common_color(colors):
    color_counts = {}

    for color in colors:
        if color is None:
            continue

        color_counts[color] = color_counts.get(color, 0) + 1

    if not color_counts:
        return None

    return max(color_counts.items(), key=lambda item: item[1])[0]


def build_blender_detail_maps(height_map, color_name_map, detail_level):
    """
    Creates reduced-detail maps for Blender.

    detail_level:
    1 = full detail, highest object count
    2 = medium detail, recommended
    3 = lower detail, faster
    4 = very low detail, fastest

    For detail level 2, each Blender cell represents a 2x2 area.
    For detail level 3, each Blender cell represents a 3x3 area.
    """
    block_size = detail_level

    blender_height = int(np.ceil(GRID_HEIGHT / block_size))
    blender_width = int(np.ceil(GRID_WIDTH / block_size))

    reduced_height_map = np.zeros((blender_height, blender_width), dtype=int)
    reduced_color_map = [[None for _ in range(blender_width)] for _ in range(blender_height)]

    for by in range(blender_height):
        for bx in range(blender_width):
            y_start = by * block_size
            y_end = min((by + 1) * block_size, GRID_HEIGHT)

            x_start = bx * block_size
            x_end = min((bx + 1) * block_size, GRID_WIDTH)

            block_heights = height_map[y_start:y_end, x_start:x_end]
            positive_heights = block_heights[block_heights > 0]

            if positive_heights.size == 0:
                continue

            # Use max height to preserve towers and silhouettes.
            height_value = int(np.max(positive_heights))
            reduced_height_map[by][bx] = height_value

            block_colors = []
            for yy in range(y_start, y_end):
                for xx in range(x_start, x_end):
                    if height_map[yy][xx] > 0:
                        block_colors.append(color_name_map[yy][xx])

            reduced_color_map[by][bx] = get_most_common_color(block_colors)

    return reduced_height_map, reduced_color_map, block_size


def add_blender_part(parts, x, y, z_plate, part_type, length, depth, height_plates, color):
    parts.append({
        "x": x,
        "y": y,
        "z_plate": z_plate,
        "part": f"{part_type} {length}x{depth}",
        "part_type": part_type,
        "width": length,
        "depth": depth,
        "height_plates": height_plates,
        "color": color,
        "rgb": LEGO_COLORS[color]
    })


def optimize_blender_row(row_cells, y, part_type, z_plate, block_size):
    optimized = []
    x = 0

    while x < len(row_cells):
        color = row_cells[x]

        if color is None:
            x += 1
            continue

        start_x = x
        run_length = 1

        while (
            x + run_length < len(row_cells)
            and row_cells[x + run_length] == color
        ):
            run_length += 1

        remaining = run_length
        current_x = start_x

        for size in [8, 6, 4, 3, 2, 1]:
            while remaining >= size:
                if part_type == "Brick":
                    height_plates = 3
                else:
                    height_plates = 1

                add_blender_part(
                    parts=optimized,
                    x=current_x * block_size,
                    y=y * block_size,
                    z_plate=z_plate,
                    part_type=part_type,
                    length=size * block_size,
                    depth=block_size,
                    height_plates=height_plates,
                    color=color
                )

                current_x += size
                remaining -= size

        x = start_x + run_length

    return optimized


def create_blender_optimized_parts(height_map, color_name_map, detail_level=2):
    """
    Converts the height map into optimized Blender parts.

    The important change:
    - detail_level reduces the Blender object count.
    - detail_level=2 means each Blender part represents a 2x2 original grid area.
    - detail_level=3 means each Blender part represents a 3x3 original grid area.
    """
    detail_level = clamp_blender_detail_level(detail_level)

    reduced_height_map, reduced_color_map, block_size = build_blender_detail_maps(
        height_map=height_map,
        color_name_map=color_name_map,
        detail_level=detail_level
    )

    blender_parts = []
    reduced_grid_height = reduced_height_map.shape[0]
    reduced_grid_width = reduced_height_map.shape[1]

    for y in range(reduced_grid_height):

        # Full brick layers.
        for z_plate in range(0, MAX_HEIGHT_PLATES, 3):
            row_cells = [None] * reduced_grid_width

            for x in range(reduced_grid_width):
                height_plates = int(reduced_height_map[y][x])
                color = reduced_color_map[y][x]

                if color is None:
                    continue

                if height_plates >= z_plate + 3:
                    row_cells[x] = color

            blender_parts.extend(
                optimize_blender_row(
                    row_cells=row_cells,
                    y=y,
                    part_type="Brick",
                    z_plate=z_plate,
                    block_size=block_size
                )
            )

        # Extra plate layers above full bricks.
        for z_plate in range(MAX_HEIGHT_PLATES):
            row_cells = [None] * reduced_grid_width

            for x in range(reduced_grid_width):
                height_plates = int(reduced_height_map[y][x])
                color = reduced_color_map[y][x]

                if color is None:
                    continue

                if height_plates <= 0:
                    continue

                full_brick_height = (height_plates // 3) * 3

                if full_brick_height <= z_plate < height_plates:
                    row_cells[x] = color

            blender_parts.extend(
                optimize_blender_row(
                    row_cells=row_cells,
                    y=y,
                    part_type="Plate",
                    z_plate=z_plate,
                    block_size=block_size
                )
            )

    return blender_parts, block_size


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/generate-lego-model")
async def generate_lego_model(data: dict):
    depth_url = data.get("depth_map_url")
    original_url = data.get("original_image_url")

    include_previews = data.get("include_previews", True)
    include_full_layout = data.get("include_full_layout", False)
    include_debug_maps = data.get("include_debug_maps", True)
    include_blender_parts = data.get("include_blender_parts", True)
    blender_detail_level = clamp_blender_detail_level(data.get("blender_detail_level", 2))

    if not depth_url:
        return {"error": "Missing depth_map_url"}

    if not original_url:
        return {"error": "Missing original_image_url"}

    depth_response = requests.get(depth_url)
    original_response = requests.get(original_url)

    if depth_response.status_code != 200:
        return {"error": "Could not download depth map"}

    if original_response.status_code != 200:
        return {"error": "Could not download original image"}

    depth_img = Image.open(BytesIO(depth_response.content)).convert("L")
    depth_img = depth_img.resize((GRID_WIDTH, GRID_HEIGHT), Image.Resampling.LANCZOS)
    depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=0.15))

    original_img = Image.open(BytesIO(original_response.content)).convert("RGB")

    original_clean = original_img.resize((GRID_WIDTH, GRID_HEIGHT), Image.Resampling.LANCZOS)

    original_color = original_clean.filter(ImageFilter.GaussianBlur(radius=0.10))
    original_color = original_color.filter(ImageFilter.EDGE_ENHANCE)
    original_color = original_color.filter(ImageFilter.SHARPEN)
    original_color = ImageEnhance.Color(original_color).enhance(1.08)
    original_color = ImageEnhance.Contrast(original_color).enhance(1.05)
    original_color = ImageEnhance.Sharpness(original_color).enhance(1.10)

    color_arr = np.array(original_color)

    (
        height_map,
        normalized_depth_arr,
        edge_map,
        sky_mask,
        dark_recess_mask,
        bright_detail_mask
    ) = build_improved_height_map(depth_img, original_clean)

    brick_layout = []
    optimized_parts = []
    parts_summary = {}
    color_summary = {}
    height_summary = {}

    color_name_map = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    for y in range(GRID_HEIGHT):
        brick_row = [None] * GRID_WIDTH
        plate_row = [None] * GRID_WIDTH

        for x in range(GRID_WIDTH):
            height_plates = int(height_map[y][x])

            if height_plates > 0:
                rgb = tuple(color_arr[y][x])
                color = closest_lego_color(rgb)

                color_name_map[y][x] = color

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

    blender_optimized_parts = []
    blender_block_size = blender_detail_level

    if include_blender_parts:
        blender_optimized_parts, blender_block_size = create_blender_optimized_parts(
            height_map=height_map,
            color_name_map=color_name_map,
            detail_level=blender_detail_level
        )

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
        "unique_colors": len(color_summary),
        "blender_optimized_part_count": len(blender_optimized_parts),
        "blender_detail_level": blender_detail_level,
        "blender_block_size": blender_block_size,
        "edge_detail_boost": EDGE_DETAIL_BOOST,
        "depth_edge_boost": DEPTH_EDGE_BOOST,
        "window_recess_amount": WINDOW_RECESS_AMOUNT,
        "bright_detail_boost": BRIGHT_DETAIL_BOOST,
        "min_depth_value": MIN_DEPTH_VALUE,
        "canny_thresholds": "80,180"
    }

    response = {
        "message": "LEGO model generated with Blender optimized parts",
        "model_stats": model_stats,
        "grid_size": {
            "width": GRID_WIDTH,
            "height": GRID_HEIGHT
        },
        "max_height_plates": MAX_HEIGHT_PLATES,
        "parts_summary": parts_summary,
        "color_summary": color_summary,
        "height_summary": height_summary,
        "optimized_parts": optimized_parts,
        "blender_render_config": {
            "unit": "studs",
            "stud_size": 1.0,
            "plate_height": 0.32,
            "brick_height": 0.96,
            "grid_width": GRID_WIDTH,
            "grid_height": GRID_HEIGHT,
            "max_height_plates": MAX_HEIGHT_PLATES,
            "part_width_axis": "x",
            "part_depth_axis": "y",
            "vertical_axis": "z",
            "blender_detail_level": blender_detail_level,
            "blender_block_size": blender_block_size
        }
    }

    if include_previews:
        response["reference_preview_base64"] = create_reference_like_preview(brick_layout)
        response["stud_preview_base64"] = create_stud_preview(brick_layout)
        response["isometric_preview_base64"] = create_isometric_preview(brick_layout)

    if include_blender_parts:
        response["blender_optimized_parts"] = blender_optimized_parts

    if include_debug_maps:
        response["debug_maps"] = {
            "normalized_depth_base64": create_debug_map_base64(normalized_depth_arr),
            "edge_map_base64": create_debug_map_base64(edge_map),
            "sky_mask_base64": create_debug_map_base64(sky_mask),
            "dark_recess_mask_base64": create_debug_map_base64(dark_recess_mask),
            "bright_detail_mask_base64": create_debug_map_base64(bright_detail_mask),
            "final_height_map_base64": create_debug_map_base64(
                (height_map / MAX_HEIGHT_PLATES * 255).astype(np.uint8)
            )
        }

    if include_full_layout:
        response["brick_layout"] = brick_layout

    return response
