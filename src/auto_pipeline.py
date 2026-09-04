"""
GeoSR - Automated Satellite Super-Resolution Pipeline
Usage:
    python src/auto_pipeline.py --preset stadium_airport
    python src/auto_pipeline.py --all-presets
    python src/auto_pipeline.py --input path/to/image.png
"""

import os
import sys
import argparse
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.processor import GeoSRProcessor

PRESETS = {
    "stadium_airport": ROOT_DIR / "data" / "presets" / "stadium_airport.png",
    "urban_core": ROOT_DIR / "data" / "presets" / "urban_core.png",
    "yamuna_river": ROOT_DIR / "data" / "presets" / "yamuna_river.png",
    "agricultural_grid": ROOT_DIR / "data" / "presets" / "agricultural_grid.png",
}


def print_banner():
    print("=" * 65)
    print("        GeoSR · Satellite Super-Resolution (DetailEDSR 4×)")
    print("=" * 65)


def process_single(processor, input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    print(f"\n[+] Processing: {input_path.name}")
    print(f"    File path: {input_path}")

    start_time = time.time()
    results = processor.enhance_file(input_path, output_dir=output_dir, save_geotiff=True)
    elapsed = time.time() - start_time

    paths = results["paths"]
    m_in = results["metrics_input"]
    m_bi = results["metrics_bicubic"]
    m_sr = results["metrics_sr"]

    lap_gain = ((m_sr["laplacian_variance"] - m_bi["laplacian_variance"]) / max(1e-3, m_bi["laplacian_variance"])) * 100.0
    grad_gain = ((m_sr["gradient_energy"] - m_bi["gradient_energy"]) / max(1e-3, m_bi["gradient_energy"])) * 100.0

    print(f"    Time taken: {elapsed:.2f}s")
    print("    Metrics Comparison:")
    print(f"      - Native Sentinel (10m):   Laplacian Var = {m_in['laplacian_variance']:6.2f} | Grad Energy = {m_in['gradient_energy']:6.2f}")
    print(f"      - Bicubic 4x Baseline:     Laplacian Var = {m_bi['laplacian_variance']:6.2f} | Grad Energy = {m_bi['gradient_energy']:6.2f}")
    print(f"      - GeoSR Enhanced 4x:       Laplacian Var = {m_sr['laplacian_variance']:6.2f} | Grad Energy = {m_sr['gradient_energy']:6.2f}")
    print(f"      -> Laplacian statistic delta: {lap_gain:+.1f}% vs Bicubic")
    print(f"      -> Gradient statistic delta:  {grad_gain:+.1f}% vs Bicubic")
    print("      Note: these are image statistics, not ground-truth accuracy metrics.")
    print("    Artifacts Saved:")
    print(f"      - High-Res PNG:    {paths['sr_png']}")
    print(f"      - Comparison View: {paths['comparison_png']}")
    if paths.get("geotiff"):
        print(f"      - GeoTIFF Raster:  {paths['geotiff']}")
    return results


def main():
    parser = argparse.ArgumentParser(description="GeoSR Automated Super-Resolution Pipeline")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input image or GeoTIFF/JP2")
    parser.add_argument("--preset", "-p", type=str, choices=list(PRESETS.keys()), default=None, help="Process a hero preset scene")
    parser.add_argument("--all-presets", action="store_true", help="Process all 4 hero preset scenes")
    parser.add_argument("--output", "-o", type=str, default=str(ROOT_DIR / "outputs" / "enhanced"), help="Output directory")

    args = parser.parse_args()
    print_banner()

    processor = GeoSRProcessor()

    if args.all_presets:
        print("\nProcessing all 4 landmark preset scenes...")
        for name, path in PRESETS.items():
            if path.exists():
                process_single(processor, path, args.output)
        print("\n[OK] All presets successfully super-resolved.")
        return

    if args.preset:
        path = PRESETS[args.preset]
        process_single(processor, path, args.output)
        print("\n[OK] Preset processing complete.")
        return

    if args.input:
        process_single(processor, args.input, args.output)
        print("\n[OK] Processing complete.")
        return

    # Default fallback: process stadium_airport
    print("No input specified. Defaulting to 'stadium_airport' preset.")
    process_single(processor, PRESETS["stadium_airport"], args.output)


if __name__ == "__main__":
    main()
