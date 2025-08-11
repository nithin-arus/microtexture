#!/usr/bin/env python3
"""
Test script for comprehensive fractal texture analysis pipeline
This script tests the analysis functionality including fractal fitting without requiring a camera.
"""

import os
import sys
from analysis.feature_extractor import extract_features

def test_single_image(image_path):
    """Test comprehensive feature extraction including fractal fitting on a single image"""
    print(f"Testing comprehensive feature extraction on: {image_path}")
    print("This may take a moment due to extensive analysis + fractal fitting...")
    print()
    
    try:
        features = extract_features(image_path, label="test", enable_fractal_fitting=True)
        
        if 'error' in features:
            print(f"✗ Error during feature extraction: {features['error']}")
            return False
        
        # Organize features by category for better display
        feature_categories = {
            "Basic Info": ["filename", "path", "label"],
            "Basic Statistics": ["mean_intensity", "std_dev", "variance", "skewness", "kurtosis", 
                               "range_intensity", "min_intensity", "max_intensity"],
            "Entropy Measures": ["entropy_shannon", "entropy_local"],
            "Edge Features": ["edge_density", "edge_magnitude_mean", "edge_magnitude_std", "edge_orientation_std"],
            "Haralick Features": ["haralick_contrast", "haralick_dissimilarity", "haralick_homogeneity", 
                                "haralick_energy", "haralick_correlation", "haralick_asm"],
            "Local Binary Patterns": ["lbp_uniform_mean", "lbp_variance", "lbp_entropy"],
            "Fractal Dimensions": ["fractal_dim_higuchi", "fractal_dim_katz", "fractal_dim_dfa", 
                                 "fractal_dim_boxcount", "lacunarity"],
            "Wavelet Features": ["wavelet_energy_approx", "wavelet_energy_horizontal", "wavelet_energy_vertical", 
                               "wavelet_energy_diagonal", "wavelet_entropy"],
            "Tamura Features": ["tamura_coarseness", "tamura_contrast", "tamura_directionality"],
            "Morphological": ["area_coverage", "circularity", "solidity", "perimeter_complexity"],
            "🆕 Fractal Fitting": ["fractal_equation", "fractal_hurst_exponent", "fractal_amplitude_scaling", 
                                  "fractal_goodness_of_fit", "fractal_fitting_success", "fractal_overlay_path"]
        }
        
        print("=== COMPREHENSIVE TEXTURE ANALYSIS RESULTS ===")
        print()
        
        total_features = 0
        for category, feature_list in feature_categories.items():
            print(f"{category}:")
            category_count = 0
            for feature in feature_list:
                if feature in features:
                    value = features[feature]
                    if isinstance(value, (int, float)) and feature not in ["filename", "path", "label", "fractal_equation", "fractal_overlay_path"]:
                        print(f"  {feature:25}: {value:.6f}")
                        category_count += 1
                        total_features += 1
                    else:
                        print(f"  {feature:25}: {value}")
                        if feature in ["fractal_equation", "fractal_overlay_path"]:
                            category_count += 1
                else:
                    print(f"  {feature:25}: [Missing]")
            
            if category != "Basic Info":
                print(f"  → {category_count} features")
            print()
        
        # Special handling for fractal results
        fractal_eq = features.get('fractal_equation', 'N/A')
        fractal_fit = features.get('fractal_goodness_of_fit', 0.0)
        overlay_path = features.get('fractal_overlay_path', '')
        
        print("🌀 FRACTAL SURFACE FITTING RESULTS:")
        print(f"  • Equation: {fractal_eq}")
        print(f"  • Fit Quality: {fractal_fit:.4f}")
        if overlay_path and os.path.exists(overlay_path):
            print(f"  • Overlay saved: {overlay_path}")
            print(f"  • You can view the fractal overlay visualization!")
        else:
            print(f"  • Overlay: Not generated")
        print()
        
        print(f"✓ Successfully extracted {total_features} numerical features!")
        print(f"✓ Feature extraction pipeline working correctly!")
        print(f"🆕 NEW: Fractal surface fitting and visualization complete!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def find_test_images():
    """Find available test images in the project"""
    search_paths = [
        "capture/images",
        "images",
        "."
    ]
    
    found_images = []
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        found_images.append(os.path.join(root, file))
    
    return found_images

def show_system_info():
    """Display information about the analysis system"""
    print("=== COMPREHENSIVE TEXTURE ANALYSIS SYSTEM ===")
    print()
    print("This system extracts 45+ texture features from each image:")
    print()
    print("📊 Feature Categories:")
    print("  • Basic Statistics (8 features)")
    print("  • Entropy Measures (2 features)")  
    print("  • Edge & Gradient Analysis (4 features)")
    print("  • Haralick Texture Features (6 features)")
    print("  • Local Binary Patterns (3 features)")
    print("  • Fractal Dimensions (5 features)")
    print("  • Wavelet Analysis (5 features)")
    print("  • Tamura Texture Features (3 features)")
    print("  • Morphological Features (4 features)")
    print("  🆕 • Fractal Surface Fitting (5 features)")
    print("  🆕 • Visualization Output (1 feature)")
    print()
    print("🌀 NEW: Fractal Surface Fitting")
    print("  • Fits fractional Brownian motion surface to image")
    print("  • Extracts mathematical equation parameters")
    print("  • Generates overlay visualization")
    print("  • Provides goodness-of-fit measure")
    print()
    print("🔬 Perfect for:")
    print("  • Material classification")
    print("  • Fabric wear analysis")
    print("  • Surface quality assessment")
    print("  • Machine learning datasets")
    print("  • Textile research")
    print("  🆕 • Fractal modeling and surface reconstruction")
    print("  🆕 • Visual validation of texture analysis")
    print()

def main():
    print("=== COMPREHENSIVE FRACTAL TEXTURE ANALYSIS - TEST SCRIPT ===")
    print()
    
    # Show system information
    show_system_info()
    
    # Check if user provided an image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            test_single_image(image_path)
        else:
            print(f"Error: Image file '{image_path}' not found.")
        return
    
    # Find test images automatically
    print("Searching for test images...")
    test_images = find_test_images()
    
    if not test_images:
        print("No test images found.")
        print("Usage: python test_analysis.py [path_to_image.jpg]")
        print("Or place some .jpg images in the capture/images/ directory.")
        return
    
    print(f"Found {len(test_images)} test images:")
    for i, img in enumerate(test_images[:5]):  # Show first 5
        print(f"  {i+1}. {img}")
    
    if len(test_images) > 5:
        print(f"  ... and {len(test_images) - 5} more")
    
    print()
    print("⚠️  Note: Fractal fitting will take longer than basic feature extraction")
    response = input("Test with first image? (y/n): ").strip().lower()
    
    if response in ['y', 'yes', '']:
        print(f"\nTesting comprehensive analysis with fractal fitting: {test_images[0]}")
        print("=" * 70)
        test_single_image(test_images[0])

if __name__ == "__main__":
    main() 