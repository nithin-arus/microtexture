#!/usr/bin/env python3
"""
Focused Fractal Overlay Generation for Pique Polo Images
========================================================

This script specifically processes pique polo images to generate fractal overlays.
It extracts fractal features and creates visualization overlays for each image.

Features:
- Processes only pique polo images
- Fits fractal surfaces using fractional Brownian motion
- Generates overlay visualizations
- Saves results in organized directory structure
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from analysis.fractal_fitting import fit_and_visualize_fractal, FractalSurfaceFitter
from preprocess.preprocess import preprocess_image

def get_pique_polo_images():
    """Get all pique polo images from the capture directory."""
    pique_dir = Path("capture/images/pique_polo")
    images = []

    if not pique_dir.exists():
        print(f"❌ Pique polo directory not found: {pique_dir}")
        return images

    for file_path in pique_dir.glob("*.jpg"):
        images.append(file_path)
    for file_path in pique_dir.glob("*.jpeg"):
        images.append(file_path)
    for file_path in pique_dir.glob("*.png"):
        images.append(file_path)

    return sorted(images)

def process_pique_fractal_overlays():
    """Process pique polo images and generate fractal overlays."""
    print("🎯 FRACTAL OVERLAY GENERATION FOR PIQUE POLO IMAGES")
    print("=" * 60)

    # Get pique polo images
    pique_images = get_pique_polo_images()

    if not pique_images:
        print("❌ No pique polo images found!")
        return 1

    print(f"📊 Found {len(pique_images)} pique polo images:")
    for img_path in pique_images:
        print(f"  • {img_path.name}")

    # Create output directories
    output_dir = Path("pique_fractal_overlays")
    output_dir.mkdir(exist_ok=True)

    # Initialize results tracking
    results = []
    successful = 0
    failed = 0

    print("\n🌀 Starting fractal analysis and overlay generation...")
    print("This may take a few minutes per image due to fractal fitting...")

    # Process each image
    for i, img_path in enumerate(pique_images, 1):
        try:
            print(f"\n🔍 Processing {i}/{len(pique_images)}: {img_path.name}...")

            # Load and preprocess image
            gray_img, binary_img = preprocess_image(str(img_path))

            # Create fractal fitter
            fitter = FractalSurfaceFitter()

            # Fit fractal surface
            print("  Fitting fractal surface...")
            fractal_params = fitter.fit_fractal_surface(gray_img)

            # Generate overlay visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            overlay_filename = f"pique_polo_{img_path.stem}_fractal_overlay_{timestamp}.png"
            overlay_path = output_dir / overlay_filename

            print("  Creating visualization overlay...")
            success = fitter.create_fractal_overlay(gray_img, str(overlay_path))

            if success:
                print(f"  ✅ Success! Overlay saved: {overlay_filename}")
                print(f"     Fractal equation: {fractal_params.get('fractal_equation', 'N/A')}")
                print(".4f")

                # Track results
                result = {
                    'filename': img_path.name,
                    'path': str(img_path),
                    'overlay_path': str(overlay_path),
                    'fractal_equation': fractal_params.get('fractal_equation', ''),
                    'hurst_exponent': fractal_params.get('hurst_exponent', 0.0),
                    'amplitude_scaling': fractal_params.get('amplitude_scaling', 0.0),
                    'goodness_of_fit': fractal_params.get('goodness_of_fit', 0.0),
                    'fitting_success': fractal_params.get('fitting_success', False)
                }
                results.append(result)
                successful += 1
            else:
                print("  ❌ Failed to create overlay")
                failed += 1

        except Exception as e:
            print(f"  ❌ Error processing {img_path.name}: {e}")
            failed += 1

    # Generate summary report
    print("\n📊 Generating summary report...")
    generate_summary_report(results, output_dir, successful, failed)

    # Final summary
    print("\n" + "="*60)
    print("🎉 FRACTAL OVERLAY GENERATION COMPLETE!")
    print("="*60)
    print("📈 SUMMARY:")
    print(f"   • Total pique polo images: {len(pique_images)}")
    print(f"   • Successfully processed: {successful}")
    if failed > 0:
        print(f"   • Failed: {failed}")
    print(f"   • Output directory: {output_dir}/")
    print(f"   • Summary report: {output_dir}/pique_fractal_summary.txt")
    print()
    print("🖼️  Fractal overlays are ready for visual inspection!")
    print("🔬 Each overlay shows the original image with fitted fractal surface")

    return 0

def generate_summary_report(results, output_dir, successful, failed):
    """Generate a comprehensive summary report."""
    report_path = output_dir / "pique_fractal_summary.txt"

    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PIQUE POLO FRACTAL OVERLAY ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")

        f.write("SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total images processed: {len(results) + failed}\n")
        f.write(f"Successfully processed: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Success rate: {(successful / (len(results) + failed)) * 100:.1f}%\n\n")

        if results:
            f.write("FRACTAL PARAMETERS SUMMARY:\n")
            f.write("-" * 30 + "\n")

            hurst_values = [r['hurst_exponent'] for r in results if r['fitting_success']]
            goodness_values = [r['goodness_of_fit'] for r in results if r['fitting_success']]

            if hurst_values:
                f.write(f"Average Hurst exponent: {np.mean(hurst_values):.4f} ± {np.std(hurst_values):.4f}\n")
                f.write(f"Hurst range: {np.min(hurst_values):.4f} - {np.max(hurst_values):.4f}\n")

            if goodness_values:
                f.write(f"Average goodness of fit: {np.mean(goodness_values):.4f} ± {np.std(goodness_values):.4f}\n")
                f.write(f"Fit quality range: {np.min(goodness_values):.4f} - {np.max(goodness_values):.4f}\n")

            f.write("\n")

        f.write("DETAILED RESULTS:\n")
        f.write("-" * 20 + "\n")
        for result in results:
            f.write(f"Image: {result['filename']}\n")
            f.write(f"  Fractal equation: {result['fractal_equation']}\n")
            f.write(f"  Goodness of fit: {result['goodness_of_fit']:.4f}\n")
            f.write(f"  Fitting success: {'Yes' if result['fitting_success'] else 'No'}\n")
            f.write(f"  Overlay: {Path(result['overlay_path']).name}\n")
            f.write("\n")

        f.write("WHAT ARE FRACTAL OVERLAYS?\n")
        f.write("-" * 30 + "\n")
        f.write("• Each overlay visualization shows your original pique polo image\n")
        f.write("• A fitted fractal surface (fractional Brownian motion) is overlaid\n")
        f.write("• The fractal equation fBm(H=xx, σ=xx) describes the surface roughness\n")
        f.write("• Higher H values indicate smoother surfaces, lower H indicate rougher\n")
        f.write("• The goodness of fit shows how well the fractal matches your image\n")
        f.write("• These overlays help validate the fractal texture analysis\n")

    print(f"✅ Summary report saved to: {report_path}")

    # Also save results as CSV for further analysis
    if results:
        csv_path = output_dir / "pique_fractal_results.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"✅ Detailed results saved to: {csv_path}")

def main():
    """Main function."""
    try:
        return process_pique_fractal_overlays()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
