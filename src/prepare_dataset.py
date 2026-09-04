"""
GeoSR - Sentinel-2 Training Dataset Preparation

Purpose:
    Convert a Sentinel-2 product into paired:
        LR (low-resolution/degraded) images
        HR (high-resolution/reference) images

The HR image is created from the original Sentinel-2 10m RGB bands.
The LR image is synthetically degraded using:
    - Gaussian blur
    - downsampling
    - sensor-like Gaussian noise
    - mild atmospheric/haze degradation

Expected Sentinel-2 RGB bands:
    B02 = Blue
    B03 = Green
    B04 = Red

Output:
    data/processed/
        train/
            LR/
            HR/
        val/
            LR/
            HR/
"""

from pathlib import Path
import argparse

import cv2
import numpy as np
import rasterio


# ============================================================
# Configuration
# ============================================================

SCALE = 4

# Tile size in the ORIGINAL / HR image.
# 64x64 pixels at Sentinel-2's 10m resolution = 640m x 640m.
HR_TILE_SIZE = 64

# Overlap between neighboring tiles.
STRIDE = 32

# Minimum amount of useful image information in a tile.
# Prevents us from creating thousands of mostly-empty patches.
MIN_VALID_RATIO = 0.90

RANDOM_SEED = 42


# ============================================================
# Sentinel-2 band discovery
# ============================================================

def find_band(product_path: Path, band_name: str) -> Path:
    """
    Recursively search for a Sentinel-2 band.

    Example:
        B02_10m.jp2
        T32T..._B02_10m.jp2
    """

    matches = list(product_path.rglob(f"*_{band_name}_10m.jp2"))

    if not matches:
        # Some products may have slightly different naming.
        matches = list(product_path.rglob(f"*{band_name}*10m*.jp2"))

    if not matches:
        raise FileNotFoundError(
            f"Could not find Sentinel-2 band {band_name} at:\n"
            f"{product_path}\n\n"
            f"Make sure this is the extracted Sentinel-2 product "
            f"and that the 10m JP2 bands exist."
        )

    if len(matches) > 1:
        print(f"\nMultiple {band_name} files found:")
        for i, match in enumerate(matches, start=1):
            print(f"  [{i}] {match}")

        print("\nUsing the first matching file.")

    return matches[0]


# ============================================================
# Reading Sentinel-2 RGB
# ============================================================

def load_rgb(product_path: Path):
    """
    Load Sentinel-2 B04/B03/B02 as an RGB image.

    Returns:
        rgb       : float32 image, shape (H, W, 3)
        transform : rasterio affine transform
        crs       : coordinate reference system
    """

    blue_path = find_band(product_path, "B02")
    green_path = find_band(product_path, "B03")
    red_path = find_band(product_path, "B04")

    print("\nFound bands:")
    print(f"  Blue  (B02): {blue_path}")
    print(f"  Green (B03): {green_path}")
    print(f"  Red   (B04): {red_path}")

    with rasterio.open(red_path) as red_src:
        red = red_src.read(1).astype(np.float32)
        transform = red_src.transform
        crs = red_src.crs

    with rasterio.open(green_path) as green_src:
        green = green_src.read(1).astype(np.float32)

    with rasterio.open(blue_path) as blue_src:
        blue = blue_src.read(1).astype(np.float32)

    if not (
        red.shape == green.shape == blue.shape
    ):
        raise ValueError(
            "RGB bands do not have identical dimensions.\n"
            f"Red:   {red.shape}\n"
            f"Green: {green.shape}\n"
            f"Blue:  {blue.shape}"
        )

    # Sentinel-2 L2A reflectance is commonly stored scaled.
    # We normalize each band independently for image processing.
    rgb = np.stack([red, green, blue], axis=-1)

    return rgb, transform, crs


# ============================================================
# Normalization
# ============================================================

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Robustly normalize a Sentinel-2 image to [0, 1].

    Percentile normalization prevents a few very bright pixels
    from dominating the entire image.
    """

    output = np.zeros_like(image, dtype=np.float32)

    for channel in range(image.shape[2]):
        band = image[:, :, channel]

        valid = np.isfinite(band) & (band > 0)

        if not np.any(valid):
            continue

        low = np.percentile(band[valid], 2)
        high = np.percentile(band[valid], 98)

        if high <= low:
            output[:, :, channel] = 0
            continue

        normalized = (band - low) / (high - low)
        normalized = np.clip(normalized, 0.0, 1.0)

        normalized[~valid] = 0

        output[:, :, channel] = normalized

    return output


# ============================================================
# Sentinel-2 specific degradation
# ============================================================

def degrade_image(
    hr: np.ndarray,
    scale: int = SCALE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Convert an HR image into a synthetic LR image.

    Degradation sequence:

        HR
         ↓
        Gaussian blur
         ↓
        downsampling
         ↓
        sensor noise
         ↓
        mild haze

    The result is still an RGB image in [0, 1].
    """

    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # --------------------------------------------------------
    # 1. Optical / motion-like blur
    # --------------------------------------------------------

    sigma = rng.uniform(0.6, 1.4)

    blurred = cv2.GaussianBlur(
        hr,
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    # --------------------------------------------------------
    # 2. Downsample
    # --------------------------------------------------------

    h, w = blurred.shape[:2]

    lr_width = w // scale
    lr_height = h // scale

    lr = cv2.resize(
        blurred,
        (lr_width, lr_height),
        interpolation=cv2.INTER_AREA,
    )

    # --------------------------------------------------------
    # 3. Sensor-like Gaussian noise
    # --------------------------------------------------------

    noise_std = rng.uniform(0.005, 0.025)

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=lr.shape,
    ).astype(np.float32)

    lr = lr + noise

    # --------------------------------------------------------
    # 4. Mild atmospheric haze
    # --------------------------------------------------------

    haze_strength = rng.uniform(0.0, 0.08)

    # Atmospheric light.
    atmospheric_light = rng.uniform(
        0.75,
        1.0,
        size=(1, 1, 3),
    ).astype(np.float32)

    lr = (
        (1.0 - haze_strength) * lr
        + haze_strength * atmospheric_light
    )

    # --------------------------------------------------------
    # Final clipping
    # --------------------------------------------------------

    lr = np.clip(lr, 0.0, 1.0)

    return lr.astype(np.float32)


# ============================================================
# Save PNG
# ============================================================

def save_rgb(path: Path, image: np.ndarray):
    """
    Save a float RGB image in [0,1] as an 8-bit PNG.
    """

    image_uint8 = np.clip(
        image * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # OpenCV expects BGR.
    image_bgr = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        str(path),
        image_bgr,
    )


# ============================================================
# Tile generation
# ============================================================

def generate_tiles(
    rgb: np.ndarray,
    output_dir: Path,
    split: str,
    start_index: int = 0,
):
    """
    Generate paired LR/HR tiles.

    HR tile:
        64 x 64

    LR tile:
        16 x 16

    because SCALE = 4.
    """

    hr_dir = output_dir / split / "HR"
    lr_dir = output_dir / split / "LR"

    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    h, w, _ = rgb.shape

    index = start_index
    generated = 0
    skipped = 0

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"\nGenerating {split} tiles...")
    print(f"Image size: {w} x {h}")
    print(f"HR tile:    {HR_TILE_SIZE} x {HR_TILE_SIZE}")
    print(f"LR tile:    {HR_TILE_SIZE // SCALE} x {HR_TILE_SIZE // SCALE}")
    print(f"Stride:     {STRIDE}")

    for y in range(0, h - HR_TILE_SIZE + 1, STRIDE):
        for x in range(0, w - HR_TILE_SIZE + 1, STRIDE):

            hr = rgb[
                y:y + HR_TILE_SIZE,
                x:x + HR_TILE_SIZE,
            ]

            # ------------------------------------------------
            # Reject mostly-invalid tiles.
            # ------------------------------------------------

            valid_pixels = np.all(
                np.isfinite(hr),
                axis=2,
            ) & np.any(
                hr > 0,
                axis=2,
            )

            valid_ratio = valid_pixels.mean()

            if valid_ratio < MIN_VALID_RATIO:
                skipped += 1
                continue

            # ------------------------------------------------
            # Generate LR version.
            # ------------------------------------------------

            lr = degrade_image(
                hr,
                scale=SCALE,
                rng=rng,
            )

            filename = f"{index:06d}.png"

            save_rgb(
                hr_dir / filename,
                hr,
            )

            save_rgb(
                lr_dir / filename,
                lr,
            )

            generated += 1
            index += 1

    print("\nTile generation complete.")
    print(f"Generated: {generated}")
    print(f"Skipped:   {skipped}")

    return generated


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Prepare GeoSR Sentinel-2 training pairs."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the extracted Sentinel-2 .SAFE directory.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Output dataset directory.",
    )

    args = parser.parse_args()

    product_path = Path(args.input)
    output_dir = Path(args.output)

    if not product_path.exists():
        raise FileNotFoundError(
            f"\nInput path does not exist:\n{product_path}\n"
        )

    print("=" * 60)
    print("GeoSR Dataset Preparation")
    print("=" * 60)

    print(f"\nInput:")
    print(product_path.resolve())

    print(f"\nOutput:")
    print(output_dir.resolve())

    # --------------------------------------------------------
    # Load Sentinel-2 RGB
    # --------------------------------------------------------

    print("\nLoading Sentinel-2 RGB bands...")

    rgb_raw, transform, crs = load_rgb(product_path)

    print(f"\nOriginal image shape: {rgb_raw.shape}")
    print(f"CRS: {crs}")

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print("\nNormalizing reflectance values...")

    rgb = normalize_image(rgb_raw)

    print(
        f"Normalized range: "
        f"{rgb.min():.4f} - {rgb.max():.4f}"
    )

    # --------------------------------------------------------
    # Generate train/validation split
    # --------------------------------------------------------

    # IMPORTANT:
    # Because this is one geographical scene, we split spatially
    # rather than randomly mixing neighboring pixels.
    #
    # Left 80%  -> training
    # Right 20% -> validation
    #
    # This is better for preventing spatial leakage.

    _, width, _ = rgb.shape

    split_column = int(width * 0.80)

    train_rgb = rgb[:, :split_column, :]
    val_rgb = rgb[:, split_column:, :]

    print("\nSpatial split:")
    print(f"Training region:   width = {train_rgb.shape[1]}")
    print(f"Validation region: width = {val_rgb.shape[1]}")

    # --------------------------------------------------------
    # Generate datasets
    # --------------------------------------------------------

    train_count = generate_tiles(
        rgb=train_rgb,
        output_dir=output_dir,
        split="train",
        start_index=0,
    )

    val_count = generate_tiles(
        rgb=val_rgb,
        output_dir=output_dir,
        split="val",
        start_index=0,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)

    print(f"\nTraining pairs:   {train_count}")
    print(f"Validation pairs: {val_count}")

    print("\nExpected structure:")

    print(
        """
data/
└── processed/
    ├── train/
    │   ├── HR/
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   └── LR/
    │       ├── 000000.png
    │       ├── 000001.png
    │       └── ...
    │
    └── val/
        ├── HR/
        │   ├── 000000.png
        │   └── ...
        └── LR/
            ├── 000000.png
            └── ...
"""
    )


if __name__ == "__main__":
    main()