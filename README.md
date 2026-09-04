# GeoSR | Satellite Imagery Super-Resolution Platform

GeoSR is an end-to-end visual-enhancement system for Sentinel-2 RGB imagery. It creates a 4X enlarged pixel grid using an EDSR model and offers a bicubic baseline for comparison. The output is AI-enhanced imagery, not verified recovery of new 2.5m ground detail.

---

## Capabilities

- **Residual reconstruction:** Uses the deployed EDSR checkpoint as a learned detail residual added once to a bicubic 4X base, with forced sharpening disabled.
- **Live Copernicus STAC Streaming:** Directly queries open Copernicus Sentinel-2 L2A catalogs and streams Cloud-Optimized GeoTIFFs (COGs) for any target city or bounding box coordinates in real time.
- **Raster-aware ingest:** Handles common RGB imagery and, where Rasterio is available, reads geospatial rasters while preserving their CRS and output transform.
- **Enterprise Web Interface:** Streamlit dashboard featuring an interactive split-screen curtain slider, digital loupe magnification (2X to 8X), synchronized dual views, and direct GeoTIFF/PNG export.

---

## Project Structure

```text
GeoSR/
├── api/
│   ├── copernicus.py           # Copernicus CDSE token and catalog query client
│   └── satellite_fetcher.py    # Live Sentinel-2 L2A STAC streaming engine
├── app/
│   └── app.py                  # Enterprise Streamlit Web Application
├── data/
│   ├── presets/                # Verified landmark scenes (400x400 crops)
│   └── B02_10m.jp2 ...         # Native Sentinel-2 L2A bands (10m)
├── models/
│   └── edsr_visual_best.pth    # Satellite DetailEDSR neural weights (3.7 MB)
├── outputs/
│   └── enhanced/               # Super-resolved 4X PNGs and comparisons
└── src/
    ├── auto_pipeline.py        # Automated CLI super-resolution runner
    ├── processor.py            # Photometric enhancement and inference engine
    └── extract_hero_presets.py # Extraction utility for landmark scenes
```

---

## Quick Start

### 1. Launch the Web Interface

```powershell
.\.venv\Scripts\streamlit run app/app.py
```
Open `http://localhost:8501` to access the platform. You can query live Sentinel-2 scenes, explore pre-cached landmark scenes, or ingest local GeoTIFFs.

### 2. Run the CLI Pipeline

Super-resolve a single landmark scene:
```powershell
.\.venv\Scripts\python.exe src/auto_pipeline.py --preset stadium_airport
```

Batch process all verified landmark scenes:
```powershell
.\.venv\Scripts\python.exe src/auto_pipeline.py --all-presets
```

Enhance any local imagery file:
```powershell
.\.venv\Scripts\python.exe src/auto_pipeline.py --input path/to/image.tif --output outputs/enhanced/
```

---

## Evaluation note

The app shows Laplacian variance and gradient energy as display statistics only. They are not accuracy metrics: sharpening can increase both without recovering correct geographic detail. A future scientific benchmark should compare outputs with registered higher-resolution reference imagery on unseen scenes using PSNR/SSIM and visual inspection.
