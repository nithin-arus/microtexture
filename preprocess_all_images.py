#!/usr/bin/env python3
"""
Comprehensive Image Preprocessing Pipeline
==========================================

This script applies comprehensive preprocessing and augmentation to all images
in the capture/images directory, generating multiple variants for robust analysis.

Preprocessing steps:
1. Basic preprocessing (grayscale, resize, threshold)
2. Geometric augmentations (rotations, flips)
3. Photometric augmentations (brightness, contrast, gamma, noise)
4. Multi-scale augmentations (zoom, crop)
5. Imaging condition simulation (lighting, blur, resolution)

Output:
- preprocessed_images/: Basic preprocessed images
- augmented_images/: All augmentation variants
- preprocessing_report.txt: Summary of processing
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from preprocess.preprocess import preprocess_image
from preprocess.augmentation import TextureAugmentation, ImagingConditionSimulator

def create_output_directories(base_dir):
    """Create organized output directory structure."""
    dirs = [
        base_dir / "preprocessed_images",
        base_dir / "augmented_images" / "geometric",
        base_dir / "augmented_images" / "photometric",
        base_dir / "augmented_images" / "multiscale",
        base_dir / "augmented_images" / "imaging_conditions"
    ]

    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs

def get_all_images(image_root):
    """Get all images organized by fabric type."""
    image_files = {}

    for item in os.listdir(image_root):
        item_path = image_root / item
        if item_path.is_dir():
            fabric_type = item
            images = []
            for file in item_path.glob("*.jpg"):
                images.append(file)
            for file in item_path.glob("*.jpeg"):
                images.append(file)
            for file in item_path.glob("*.png"):
                images.append(file)

            if images:
                image_files[fabric_type] = images

    return image_files

def save_image(image, output_path, normalize=True):
    """Save image with proper normalization."""
    if normalize and image.max() > 1.0:
        image = (image * 255).astype(np.uint8)
    elif not normalize and image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    cv2.imwrite(str(output_path), image)

def apply_basic_preprocessing(image_path, output_dir, fabric_type):
    """Apply basic preprocessing to a single image."""
    try:
        # Load and preprocess image
        gray, binary = preprocess_image(str(image_path))

        # Save preprocessed images
        base_name = image_path.stem

        gray_path = output_dir / "preprocessed_images" / f"{fabric_type}_{base_name}_gray.jpg"
        binary_path = output_dir / "preprocessed_images" / f"{fabric_type}_{base_name}_binary.jpg"

        cv2.imwrite(str(gray_path), gray)
        cv2.imwrite(str(binary_path), binary)

        return True, gray, binary

    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return False, None, None

def apply_comprehensive_augmentation(image, fabric_type, base_name, output_dir):
    """Apply comprehensive augmentation to a single image."""
    try:
        # Initialize augmenters
        augmenter = TextureAugmentation(random_state=42)
        simulator = ImagingConditionSimulator(random_state=42)

        augmented_count = 0

        # Convert to float for augmentation
        if image.max() > 1.0:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)

        # Geometric augmentations
        print(f"  Applying geometric augmentations...")
        geometric_augs = augmenter.geometric_augmentation(image_float)
        for i, aug_img in enumerate(geometric_augs[1:], 1):  # Skip original
            output_path = output_dir / "augmented_images" / "geometric" / f"{fabric_type}_{base_name}_geo_{i}.jpg"
            save_image(aug_img, output_path)
            augmented_count += 1

        # Photometric augmentations
        print(f"  Applying photometric augmentations...")
        photometric_augs = augmenter.photometric_augmentation(image_float)
        for i, aug_img in enumerate(photometric_augs):
            output_path = output_dir / "augmented_images" / "photometric" / f"{fabric_type}_{base_name}_photo_{i}.jpg"
            save_image(aug_img, output_path)
            augmented_count += 1

        # Multi-scale augmentations
        print(f"  Applying multi-scale augmentations...")
        multiscale_augs = augmenter.multi_scale_augmentation(image_float)
        for i, aug_img in enumerate(multiscale_augs):
            output_path = output_dir / "augmented_images" / "multiscale" / f"{fabric_type}_{base_name}_scale_{i}.jpg"
            save_image(aug_img, output_path)
            augmented_count += 1

        # Imaging condition simulation
        print(f"  Applying imaging condition simulation...")
        imaging_variants = simulator.apply_comprehensive_simulation(image_float)
        for variant_name, aug_img in imaging_variants.items():
            if variant_name != 'original':  # Skip original
                output_path = output_dir / "augmented_images" / "imaging_conditions" / f"{fabric_type}_{base_name}_{variant_name}.jpg"
                save_image(aug_img, output_path)
                augmented_count += 1

        return augmented_count

    except Exception as e:
        print(f"Error augmenting {fabric_type}_{base_name}: {e}")
        return 0

def generate_preprocessing_report(image_stats, output_dir):
    """Generate a comprehensive preprocessing report."""
    report_path = output_dir / "preprocessing_report.txt"

    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPREHENSIVE IMAGE PREPROCESSING REPORT\n")
        f.write("="*60 + "\n\n")

        f.write("SUMMARY:\n")
        f.write("-" * 20 + "\n")
        total_original = sum(stats['original_count'] for stats in image_stats.values())
        total_preprocessed = sum(stats['preprocessed_count'] for stats in image_stats.values())
        total_augmented = sum(stats['augmented_count'] for stats in image_stats.values())

        f.write(f"Total original images: {total_original}\n")
        f.write(f"Total preprocessed images: {total_preprocessed}\n")
        f.write(f"Total augmented variants: {total_augmented}\n")
        f.write(f"Total images after preprocessing: {total_preprocessed + total_augmented}\n\n")

        f.write("FABRIC TYPE BREAKDOWN:\n")
        f.write("-" * 25 + "\n")
        for fabric_type, stats in image_stats.items():
            f.write(f"{fabric_type}:\n")
            f.write(f"  Original: {stats['original_count']}\n")
            f.write(f"  Preprocessed: {stats['preprocessed_count']}\n")
            f.write(f"  Augmented: {stats['augmented_count']}\n")
            f.write(f"  Total: {stats['preprocessed_count'] + stats['augmented_count']}\n\n")

        f.write("PREPROCESSING STEPS APPLIED:\n")
        f.write("-" * 30 + "\n")
        f.write("1. Basic Preprocessing:\n")
        f.write("   - Grayscale conversion\n")
        f.write("   - Resize to 512x512\n")
        f.write("   - Binary thresholding\n\n")

        f.write("2. Geometric Augmentations:\n")
        f.write("   - Rotations (90°, 180°, 270°)\n")
        f.write("   - Horizontal and vertical flips\n")
        f.write("   - Diagonal flips\n\n")

        f.write("3. Photometric Augmentations:\n")
        f.write("   - Brightness adjustments\n")
        f.write("   - Contrast adjustments\n")
        f.write("   - Gamma corrections\n")
        f.write("   - Gaussian noise addition\n\n")

        f.write("4. Multi-scale Augmentations:\n")
        f.write("   - Multiple zoom levels\n")
        f.write("   - Random cropping\n")
        f.write("   - Scale-invariant transformations\n\n")

        f.write("5. Imaging Condition Simulation:\n")
        f.write("   - Low/bright lighting\n")
        f.write("   - Directional lighting\n")
        f.write("   - Motion blur\n")
        f.write("   - Depth of field\n")
        f.write("   - Resolution changes\n\n")

        f.write("OUTPUT DIRECTORY STRUCTURE:\n")
        f.write("-" * 30 + "\n")
        f.write("preprocessed_images/:\n")
        f.write("  - {fabric_type}_{image_name}_gray.jpg\n")
        f.write("  - {fabric_type}_{image_name}_binary.jpg\n\n")

        f.write("augmented_images/geometric/:\n")
        f.write("  - {fabric_type}_{image_name}_geo_{variant}.jpg\n\n")

        f.write("augmented_images/photometric/:\n")
        f.write("  - {fabric_type}_{image_name}_photo_{variant}.jpg\n\n")

        f.write("augmented_images/multiscale/:\n")
        f.write("  - {fabric_type}_{image_name}_scale_{variant}.jpg\n\n")

        f.write("augmented_images/imaging_conditions/:\n")
        f.write("  - {fabric_type}_{image_name}_{condition}.jpg\n\n")

    print(f"✅ Preprocessing report saved to: {report_path}")

def main():
    """Main preprocessing pipeline."""
    print("🎯 COMPREHENSIVE IMAGE PREPROCESSING PIPELINE")
    print("=" * 60)

    # Setup paths
    image_root = Path("capture/images")
    output_dir = Path("preprocessed_output")

    if not image_root.exists():
        print(f"❌ Image directory not found: {image_root}")
        return 1

    # Create output directories
    print("📁 Creating output directories...")
    output_dirs = create_output_directories(output_dir)

    # Get all images
    print("🔍 Scanning for images...")
    image_files = get_all_images(image_root)

    if not image_files:
        print("❌ No images found in capture/images/")
        return 1

    print(f"📊 Found images for {len(image_files)} fabric types:")
    for fabric_type, images in image_files.items():
        print(f"  • {fabric_type}: {len(images)} images")

    # Initialize statistics
    image_stats = {}
    total_processed = 0
    total_augmented = 0

    # Process each fabric type
    for fabric_type, images in image_files.items():
        print(f"\n🎨 Processing {fabric_type} ({len(images)} images)...")

        fabric_stats = {
            'original_count': len(images),
            'preprocessed_count': 0,
            'augmented_count': 0
        }

        # Process each image
        for image_path in tqdm(images, desc=f"Processing {fabric_type}"):
            # Basic preprocessing
            success, gray_img, binary_img = apply_basic_preprocessing(
                image_path, output_dir, fabric_type
            )

            if success:
                fabric_stats['preprocessed_count'] += 2  # gray + binary

                # Comprehensive augmentation
                aug_count = apply_comprehensive_augmentation(
                    gray_img, fabric_type, image_path.stem, output_dir
                )
                fabric_stats['augmented_count'] += aug_count

        image_stats[fabric_type] = fabric_stats
        total_processed += fabric_stats['preprocessed_count']
        total_augmented += fabric_stats['augmented_count']

        print(f"✅ {fabric_type}: {fabric_stats['preprocessed_count']} preprocessed, {fabric_stats['augmented_count']} augmented")

    # Generate report
    print("\n📊 Generating preprocessing report...")
    generate_preprocessing_report(image_stats, output_dir)

    # Final summary
    print("\n" + "="*60)
    print("🎉 PREPROCESSING COMPLETE!")
    print("="*60)
    print("📈 SUMMARY:")
    print(f"   • Original images: {sum(stats['original_count'] for stats in image_stats.values())}")
    print(f"   • Preprocessed images: {total_processed}")
    print(f"   • Augmented variants: {total_augmented}")
    print(f"   • Total processed: {total_processed + total_augmented}")
    print()
    print("📁 Output directories:")
    print(f"   • Basic preprocessing: {output_dir}/preprocessed_images/")
    print(f"   • Augmentations: {output_dir}/augmented_images/")
    print(f"   • Report: {output_dir}/preprocessing_report.txt")
    print()
    print("🚀 Ready for feature extraction and machine learning!")

    return 0

if __name__ == "__main__":
    exit(main())

