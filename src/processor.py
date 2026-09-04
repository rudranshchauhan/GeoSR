import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "edsr_visual_best.pth"


class ResidualBlock(nn.Module):
    def __init__(self, channels=64, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.res_scale = res_scale

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return x + y * self.res_scale


class DetailEDSR(nn.Module):
    def __init__(self, features=64, num_blocks=8, res_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(3, features, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(features, res_scale) for _ in range(num_blocks)])
        self.body_conv = nn.Conv2d(features, features, 3, padding=1)

        self.up1 = nn.Sequential(
            nn.Conv2d(features, features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(features, features * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.tail = nn.Conv2d(features, 3, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        residual = x
        x = self.body(x)
        x = self.body_conv(x)
        x = x + residual
        x = self.up1(x)
        x = self.up2(x)
        return self.tail(x)


def load_detail_edsr(model_path=DEFAULT_MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)
    if not model_path.exists():
        alt_path = ROOT_DIR / "models" / "edsr_detail_best.pth"
        if alt_path.exists():
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {model_path} or {alt_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    features = checkpoint.get("features", 64) if isinstance(checkpoint, dict) else 64
    num_blocks = checkpoint.get("num_blocks", 8) if isinstance(checkpoint, dict) else 8

    model = DetailEDSR(features=features, num_blocks=num_blocks)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def normalize_satellite_band(band_data, p_low=2.0, p_high=98.0):
    """Percentile normalization for 12-bit raw Sentinel bands."""
    band_data = np.nan_to_num(band_data).astype(np.float32)
    low = np.percentile(band_data, p_low)
    high = np.percentile(band_data, p_high)
    if high <= low:
        return np.zeros_like(band_data, dtype=np.float32)
    return np.clip((band_data - low) / (high - low), 0.0, 1.0)


def compute_sharpness_metrics(image_np):
    """Objective sharpness metrics: Laplacian Variance and Gradient Energy."""
    if image_np.ndim == 3:
        gray = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
    else:
        gray = image_np

    pad = np.pad(gray, 1, mode="reflect")
    lap = (
        pad[:-2, 1:-1]
        + pad[2:, 1:-1]
        + pad[1:-1, :-2]
        + pad[1:-1, 2:]
        - 4.0 * pad[1:-1, 1:-1]
    )
    lap_var = float(np.var(lap) * 10000.0)

    gx = pad[1:-1, 2:] - pad[1:-1, :-2]
    gy = pad[2:, 1:-1] - pad[:-2, 1:-1]
    energy = float(np.mean(gx**2 + gy**2) * 1000.0)

    return {
        "laplacian_variance": round(lap_var, 2),
        "gradient_energy": round(energy, 2)
    }


def apply_photometric_enhancement(rgb_np, strength="none"):
    """
    Optional, conservative display-only enhancement.

    The neural model output is returned unchanged by default.  Do not apply
    contrast or sharpening as part of super-resolution: those operations can
    manufacture edges and make a visually worse image look "sharper".

    Stages:
      1. Atmospheric dehazing via strong CLAHE in LAB lightness channel
      2. Multi-scale unsharp masking at fine + medium + coarse radii
      3. Bilateral edge residual injection for structural crispness
      4. Vivid color correction: saturation boost + warm tone shift
      5. Final global contrast stretch
    """
    if strength == "none":
        return np.clip(rgb_np, 0.0, 1.0)

    # A deliberately mild option for presentation only.  It must never be
    # described as recovered spatial information.
    p = {"clahe_clip": 1.3, "usm_fine": 0.18, "usm_med": 0.0,
         "usm_coarse": 0.0, "edge_w": 0.0, "sat_boost": 1.03,
         "contrast": 1.03}

    img = np.clip(rgb_np, 0.0, 1.0)
    uint8 = (img * 255.0).astype(np.uint8)

    # --- Stage 1: Atmospheric Dehazing via CLAHE in LAB ---
    lab = cv2.cvtColor(uint8, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    dehazed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    # --- Stage 2: Multi-Scale Unsharp Masking ---
    # Fine detail (sigma=0.8): micro-textures, rooftop edges
    blur_fine = cv2.GaussianBlur(dehazed, (0, 0), 0.8)
    hf_fine = dehazed - blur_fine
    # Medium detail (sigma=2.0): road networks, field boundaries
    blur_med = cv2.GaussianBlur(dehazed, (0, 0), 2.0)
    hf_med = dehazed - blur_med
    # Coarse detail (sigma=5.0): large structures, river banks
    blur_coarse = cv2.GaussianBlur(dehazed, (0, 0), 5.0)
    hf_coarse = dehazed - blur_coarse

    sharpened = dehazed + p["usm_fine"] * hf_fine + p["usm_med"] * hf_med + p["usm_coarse"] * hf_coarse
    sharpened = np.clip(sharpened, 0.0, 1.0)

    # --- Stage 3: Bilateral Edge Residual Injection ---
    sharp_uint8 = (sharpened * 255.0).astype(np.uint8)
    bilateral = cv2.bilateralFilter(sharp_uint8, d=7, sigmaColor=35, sigmaSpace=35).astype(np.float32) / 255.0
    edge_residual = sharpened - bilateral
    edge_enhanced = np.clip(sharpened + p["edge_w"] * edge_residual, 0.0, 1.0)

    # --- Stage 4: Vivid Color Correction ---
    hsv = cv2.cvtColor((edge_enhanced * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p["sat_boost"], 0, 255)
    # Slight value channel boost for brightness
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    vivid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # --- Stage 5: Global Contrast Stretch ---
    mean = np.mean(vivid)
    final = np.clip((vivid - mean) * p["contrast"] + mean, 0.0, 1.0)

    return final


class GeoSRProcessor:
    """Enterprise Satellite Super-Resolution Engine."""

    def __init__(self, model=None, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model is None:
            self.model = load_detail_edsr(device=self.device)
        else:
            self.model = model.to(self.device)

        self.scale = 4

    def enhance_rgb_array(self, rgb_array, strength="none"):
        """Enhance an RGB numpy array (H, W, 3) in [0, 1]."""
        tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            base = F.interpolate(tensor, scale_factor=self.scale, mode="bicubic", align_corners=False)
            # The deployed checkpoint predicts a detail residual. Add it
            # exactly once to the bicubic base. The prior degradation came
            # from extreme post-processing after this reconstruction.
            sr = torch.clamp(base + self.model(tensor), 0.0, 1.0)

        bicubic_np = np.clip(base.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)
        sr_raw_np = np.clip(sr.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0.0, 1.0)

        # Cosmetic enhancement is opt-in and is off for all saved outputs.
        sr_np = apply_photometric_enhancement(sr_raw_np, strength=strength)

        metrics_in = compute_sharpness_metrics(rgb_array)
        metrics_bi = compute_sharpness_metrics(bicubic_np)
        metrics_sr = compute_sharpness_metrics(sr_np)

        return {
            "sr": sr_np,
            "bicubic": bicubic_np,
            "metrics_input": metrics_in,
            "metrics_bicubic": metrics_bi,
            "metrics_sr": metrics_sr
        }

    def enhance_file(self, file_path, output_dir=None, save_geotiff=True, strength="none"):
        file_path = Path(file_path)
        if output_dir is None:
            output_dir = ROOT_DIR / "outputs" / "enhanced"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        is_geotiff = False
        crs = None
        transform = None

        if HAS_RASTERIO and file_path.suffix.lower() in [".jp2", ".tif", ".tiff"]:
            try:
                with rasterio.open(file_path) as src:
                    crs = src.crs
                    transform = src.transform
                    if src.count >= 3:
                        bands = [src.read(i + 1) for i in range(3)]
                        rgb = np.dstack([normalize_satellite_band(b) for b in bands])
                    else:
                        single = src.read(1)
                        norm = normalize_satellite_band(single)
                        rgb = np.dstack([norm, norm, norm])
                    is_geotiff = True
            except Exception as e:
                print(f"Rasterio read error: {e}. Using PIL.")

        if not is_geotiff:
            pil_img = Image.open(file_path).convert("RGB")
            rgb = np.asarray(pil_img, dtype=np.float32) / 255.0

        results = self.enhance_rgb_array(rgb, strength=strength)
        sr_np = results["sr"]
        bicubic_np = results["bicubic"]

        base_stem = file_path.stem
        # Keep the broken legacy output separate so an old cache cannot be
        # silently displayed as a corrected result.
        sr_png_path = output_dir / f"{base_stem}_geosr_residual_4x.png"
        bicubic_png_path = output_dir / f"{base_stem}_bicubic_4x.png"
        comparison_png_path = output_dir / f"{base_stem}_comparison.png"

        sr_pil = Image.fromarray((sr_np * 255.0).astype(np.uint8))
        bicubic_pil = Image.fromarray((bicubic_np * 255.0).astype(np.uint8))
        sr_pil.save(sr_png_path)
        bicubic_pil.save(bicubic_png_path)

        comp = Image.new("RGB", (sr_pil.width + bicubic_pil.width, sr_pil.height))
        comp.paste(bicubic_pil, (0, 0))
        comp.paste(sr_pil, (bicubic_pil.width, 0))
        comp.save(comparison_png_path)

        geotiff_path = None
        if HAS_RASTERIO and is_geotiff and save_geotiff and crs is not None and transform is not None:
            new_transform = transform * Affine.scale(1.0 / self.scale, 1.0 / self.scale)
            geotiff_path = output_dir / f"{base_stem}_geosr_residual_4x.tif"
            sr_uint8 = (sr_np * 255.0).astype(np.uint8)
            with rasterio.open(
                geotiff_path,
                "w",
                driver="GTiff",
                height=sr_uint8.shape[0],
                width=sr_uint8.shape[1],
                count=3,
                dtype="uint8",
                crs=crs,
                transform=new_transform,
            ) as dst:
                for i in range(3):
                    dst.write(sr_uint8[:, :, i], i + 1)

        results["paths"] = {
            "sr_png": str(sr_png_path),
            "bicubic_png": str(bicubic_png_path),
            "comparison_png": str(comparison_png_path),
            "geotiff": str(geotiff_path) if geotiff_path else None
        }
        return results
