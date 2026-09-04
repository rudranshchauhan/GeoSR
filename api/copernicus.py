import os
from pathlib import Path
import requests

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"

CLIENT_ID = "cdse-public"


def get_token(username, password):
    """Authenticate with Copernicus Data Space Ecosystem and retrieve access token."""
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "username": username,
            "password": password,
            "grant_type": "password"
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["access_token"]


def search_sentinel2(
    token,
    start_date,
    end_date,
    cloud_cover=20,
    top=5
):
    """Search for Sentinel-2 L2A/L1C products within date and cloud cover thresholds."""
    params = {
        "$filter": (
            "Collection/Name eq 'SENTINEL-2' and "
            f"ContentDate/Start ge {start_date}T00:00:00.000Z and "
            f"ContentDate/Start lt {end_date}T23:59:59.999Z and "
            f"Attributes/OData.CSC.DoubleAttribute/any("
            "att:att/Name eq 'cloudCover' and "
            f"att/att/Value le {cloud_cover})"
        ),
        "$top": top
    }

    response = requests.get(
        CATALOG_URL,
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )

    response.raise_for_status()
    return response.json().get("value", [])


def download_product(token, product_id, output_zip_path, chunk_size=1024 * 1024):
    """Stream download a Sentinel-2 product from CDSE zipper endpoint."""
    output_zip_path = Path(output_zip_path)
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{DOWNLOAD_URL}({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"Downloading Sentinel product {product_id} to {output_zip_path}...")
    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(output_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100.0
                        print(f"\rProgress: {pct:5.1f}% ({downloaded / (1024*1024):.1f} MB)", end="", flush=True)

    print("\nDownload complete.")
    return output_zip_path


if __name__ == "__main__":
    print("Copernicus API module ready.")
    print("Set your CDSE username/password before using get_token().")
