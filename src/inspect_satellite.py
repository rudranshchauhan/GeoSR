from pathlib import Path

import rasterio
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

B02 = DATA_DIR / "B02_10m.jp2"
B03 = DATA_DIR / "B03_10m.jp2"
B04 = DATA_DIR / "B04_10m.jp2"
B08 = DATA_DIR / "B08_10m.jp2"


def load_band(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

    return data


def normalize(image):
    low = np.percentile(image, 2)
    high = np.percentile(image, 98)

    image = np.clip(image, low, high)
    image = (image - low) / (high - low)

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Sentinel-2 bands...")

    red = load_band(B04)
    green = load_band(B03)
    blue = load_band(B02)
    nir = load_band(B08)

    print("Band shapes:")
    print(f"B02 (Blue) : {blue.shape}")
    print(f"B03 (Green): {green.shape}")
    print(f"B04 (Red)  : {red.shape}")
    print(f"B08 (NIR)  : {nir.shape}")

    blue = normalize(blue)
    green = normalize(green)
    red = normalize(red)
    nir = normalize(nir)

    true_color = np.dstack((red, green, blue))
    false_color = np.dstack((nir, red, green))

    plt.figure(figsize=(10, 8))
    plt.imshow(true_color)
    plt.title("Sentinel-2 True Color")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "true_color.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(false_color)
    plt.title("Sentinel-2 False Color")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "false_color.png", dpi=150)
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.imshow(nir, cmap="gray")
    plt.title("Sentinel-2 NIR (B08)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nir.png", dpi=150)
    plt.show()

    print()
    print("Saved:")
    print(OUTPUT_DIR / "true_color.png")
    print(OUTPUT_DIR / "false_color.png")
    print(OUTPUT_DIR / "nir.png")


if __name__ == "__main__":
    main()