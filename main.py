from fastapi import FastAPI
import requests
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/generate-lego-model")
async def generate_lego_model(data: dict):
    depth_url = data.get("depth_map_url")

    depth_img = Image.open(BytesIO(requests.get(depth_url).content)).convert("L")
    depth_img = depth_img.resize((48, 32))

    arr = np.array(depth_img)
    height_map = np.round((arr / 255) * 12).astype(int)

    return {
        "message": "LEGO model generated",
        "height_map": height_map.tolist()
    }
