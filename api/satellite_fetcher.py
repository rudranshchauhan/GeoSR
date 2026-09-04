"""
GeoSR - Live Copernicus / Sentinel-2 STAC Satellite Ingestion Engine
Streams Sentinel-2 L2A Cloud-Optimized GeoTIFFs (COGs) directly by coordinates or city.
"""

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_geom

STAC_ENDPOINT = "https://earth-search.aws.element84.com/v1/search"

CITY_COORDINATES = {
    "New Delhi (Central / Safdarjung)": {"lat": 28.585, "lon": 77.205},
    "Dubai (Downtown / Coast)": {"lat": 25.204, "lon": 55.270},
    "Cairo (Giza / Nile River)": {"lat": 29.980, "lon": 31.140},
    "San Francisco (Bay & Bridges)": {"lat": 37.780, "lon": -122.410},
    "Singapore (Marina Bay & Port)": {"lat": 1.285, "lon": 103.855},
    "Paris (Seine & Urban Center)": {"lat": 48.856, "lon": 2.352},
}


def query_sentinel2_stac(lon, lat, buffer_deg=0.04, max_cloud=10):
    """
    Queries open Copernicus Sentinel-2 L2A STAC catalog for a bounding box.
    Returns the latest cloud-free item metadata.
    """
    bbox = [lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg]
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=730)
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": 3
    }

    try:
        response = requests.post(STAC_ENDPOINT, json=payload, timeout=12)
        response.raise_for_status()
        data = response.json()
        features = data.get("features", [])
        if not features:
            # Retry with higher cloud cover tolerance if needed
            payload["query"]["eo:cloud_cover"]["lt"] = 25
            r2 = requests.post(STAC_ENDPOINT, json=payload, timeout=12)
            features = r2.json().get("features", [])

        if not features:
            raise RuntimeError(f"No clear Sentinel-2 scenes found for coordinates ({lat:.3f}, {lon:.3f}).")

        item = features[0]
        visual_href = item["assets"].get("visual", {}).get("href")
        if not visual_href:
            # Fallback to red band asset directory
            visual_href = item["assets"].get("red", {}).get("href")

        return {
            "id": item["id"],
            "datetime": item["properties"].get("datetime", "Unknown"),
            "cloud_cover": round(float(item["properties"].get("eo:cloud_cover", 0.0)), 2),
            "platform": item["properties"].get("platform", "Sentinel-2"),
            "visual_url": visual_href,
            "bbox": bbox
        }
    except Exception as e:
        raise RuntimeError(f"STAC API query error: {e}")


def stream_sentinel_crop(visual_url, crop_size=400):
    """
    Streams a crop_size x crop_size window directly from the remote Cloud-Optimized GeoTIFF.
    """
    try:
        with rasterio.open(visual_url) as src:
            w = min(crop_size, src.width)
            h = min(crop_size, src.height)
            cx = src.width // 2
            cy = src.height // 2
            win = Window(cx - w // 2, cy - h // 2, w, h)
            data = src.read(window=win)

            # Data shape: (bands, H, W)
            if data.shape[0] >= 3:
                rgb = data[:3].transpose(1, 2, 0)
            else:
                rgb = np.dstack([data[0], data[0], data[0]])

            # Ensure valid uint8
            if rgb.max() > 255 or rgb.dtype != np.uint8:
                p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
                rgb = np.clip((rgb - p2) / max(1e-5, p98 - p2), 0, 1) * 255.0
                rgb = rgb.astype(np.uint8)

            img = Image.fromarray(rgb)
            return img
    except Exception as e:
        raise RuntimeError(f"Failed to stream Sentinel-2 COG: {e}")
