import os
import csv
import sys
from datetime import datetime
from analysis.feature_extractor import extract_features, batch_extract_features

# Import capture only if running on Raspberry Pi
try:
    from capture.capture_image import capture_image, capture_multiple_images
    RASPBERRY_PI_MODE = True
except ImportError:
    print("Warning: picamera2 not found. Running in analysis-only mode.")
    RASPBERRY_PI_MODE = False

IMAGE_ROOT = "capture/images"
CSV_PATH = "data/features.csv"

# Complete list of all features that will be extracted (now with fractal fitting)
CSV_HEADERS = [
    # Basic info
    "filename", "path", "label",
    
    # Basic statistical features
    "mean_intensity", "std_dev", "variance", "skewness", "kurtosis", 
    "range_intensity", "min_intensity", "max_intensity",
    
    # Entropy measures
    "entropy_shannon", "entropy_local",
    
    # Edge features
    "edge_density", "edge_magnitude_mean", "edge_magnitude_std", "edge_orientation_std",
    
    # Haralick texture features (GLCM-based)
    "haralick_contrast", "haralick_dissimilarity", "haralick_homogeneity", 
    "haralick_energy", "haralick_correlation", "haralick_asm",
    
    # Local Binary Pattern features
    "lbp_uniform_mean", "lbp_variance", "lbp_entropy",
    
    # Fractal dimension measures
    "fractal_dim_higuchi", "fractal_dim_katz", "fractal_dim_dfa", 
    "fractal_dim_boxcount", "lacunarity",
    
    # Wavelet features
    "wavelet_energy_approx", "wavelet_energy_horizontal", "wavelet_energy_vertical", 
    "wavelet_energy_diagonal", "wavelet_entropy",
    
    # Tamura texture features
    "tamura_coarseness", "tamura_contrast", "tamura_directionality",
    
    # Morphological features
    "area_coverage", "circularity", "solidity", "perimeter_complexity",
    
    # Fractal surface fitting parameters
    "fractal_equation", "fractal_hurst_exponent", "fractal_amplitude_scaling", 
    "fractal_goodness_of_fit", "fractal_fitting_success", "fractal_overlay_path"
]

def write_header_if_needed(csv_path):
    """Create CSV file with comprehensive headers if it doesn't exist"""
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"Created new CSV file with {len(CSV_HEADERS)} feature columns")
        print("📈 NEW: Now includes fractal surface fitting and overlay visualizations!")

def append_features_to_csv(csv_path, features):
    """Append comprehensive feature dictionary to CSV file"""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Create row with all features in the correct order
        row = []
        for header in CSV_HEADERS:
            value = features.get(header, 0.0)  # Default to 0.0 if feature missing
            row.append(value)
        
        writer.writerow(row)

def get_all_images(folder):
    """Get all images from folder structure organized by labels"""
    image_files = []
    if not os.path.exists(folder):
        return image_files
        
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            # This is a label folder
            label = item
            for file in os.listdir(item_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append((os.path.join(item_path, file), label))
        elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Image in root folder
            image_files.append((item_path, "unknown"))
    return image_files

def capture_and_analyze_workflow():
    """Interactive workflow for capturing images and analyzing them"""
    if not RASPBERRY_PI_MODE:
        print("Error: Cannot capture images. picamera2 not available.")
        return
    
    print("\n=== COMPREHENSIVE FRACTAL TEXTURE ANALYSIS - CAPTURE & ANALYZE ===")
    print("This will capture images and automatically extract ALL texture features.")
    print(f"Total features extracted per image: {len(CSV_HEADERS) - 3}")  # Exclude filename, path, label
    print("🆕 NEW: Now includes fractal surface fitting and overlay visualizations!")
    print()
    
    # Get capture parameters
    label = input("Enter material label (e.g., 'cotton', 'denim', 'polyester'): ").strip()
    if not label:
        label = "sample"
    
    try:
        count = int(input(f"Number of images to capture for '{label}' (default 5): ") or "5")
    except ValueError:
        count = 5
    
    delay = 2  # seconds between captures
    
    print(f"\nCapturing {count} images of '{label}' with {delay}s delay between shots...")
    print("Make sure your sample is properly positioned under the camera.")
    print(f"Each image will be analyzed for {len(CSV_HEADERS) - 3} texture features!")
    print("🔬 Including fractal surface fitting and overlay generation...")
    print()
    
    input("Press Enter to start capturing...")
    
    # Capture images
    captured_files = capture_multiple_images(label, count, delay)
    
    if not captured_files:
        print("No images were captured. Exiting.")
        return
    
    print(f"\n✓ Successfully captured {len(captured_files)} images")
    print("Now extracting comprehensive texture features + fractal fitting...")
    print("This may take longer due to fractal surface fitting and visualization...")
    
    # Extract features for captured images
    write_header_if_needed(CSV_PATH)
    
    for i, img_path in enumerate(captured_files, 1):
        try:
            print(f"\nProcessing {i}/{len(captured_files)}: {os.path.basename(img_path)}...")
            features = extract_features(img_path, label, enable_fractal_fitting=True)
            
            # Check if extraction was successful
            if 'error' in features:
                print(f"  ✗ Error processing image: {features['error']}")
            else:
                append_features_to_csv(CSV_PATH, features)
                num_features = len([k for k in features.keys() if k not in ['filename', 'path', 'label']])
                fractal_eq = features.get('fractal_equation', 'N/A')
                overlay_path = features.get('fractal_overlay_path', '')
                
                print(f"  ✓ Extracted {num_features} features and saved to CSV")
                print(f"  🌀 Fractal fitted: {fractal_eq}")
                if overlay_path:
                    print(f"  🖼️  Fractal overlay saved: {os.path.basename(overlay_path)}")
                
        except Exception as e:
            print(f"  ✗ Error processing {img_path}: {e}")
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"✓ All features saved to: {CSV_PATH}")
    print(f"✓ Images saved in: capture/images/{label}/")
    print(f"✓ Fractal overlays saved in: fractal_overlays/")
    print(f"✓ Ready for machine learning and research analysis!")
    print(f"🆕 NEW: Each row now includes fitted fractal equation and overlay image path!")

def analyze_existing_images():
    """Analyze all existing images in the capture/images folder"""
    print("\n=== ANALYZING EXISTING IMAGES WITH COMPREHENSIVE FEATURES + FRACTAL FITTING ===")
    print(f"Each image will be analyzed for {len(CSV_HEADERS) - 3} texture features")
    print("🆕 Including fractal surface fitting and overlay generation!")
    
    write_header_if_needed(CSV_PATH)
    images = get_all_images(IMAGE_ROOT)
    
    if not images:
        print(f"No images found in {IMAGE_ROOT}")
        return
    
    print(f"Found {len(images)} images to process.")
    print("Starting comprehensive texture analysis with fractal fitting...")
    print("⏱️  This will take longer due to fractal surface fitting...")
    
    successful = 0
    failed = 0
    
    for i, (img_path, label) in enumerate(images, 1):
        try:
            print(f"\nProcessing {i}/{len(images)}: {os.path.basename(img_path)} (label: {label})...")
            features = extract_features(img_path, label, enable_fractal_fitting=True)
            
            # Check if extraction was successful
            if 'error' in features:
                print(f"  ✗ Error: {features['error']}")
                failed += 1
            else:
                append_features_to_csv(CSV_PATH, features)
                num_features = len([k for k in features.keys() if k not in ['filename', 'path', 'label']])
                fractal_eq = features.get('fractal_equation', 'N/A')
                overlay_path = features.get('fractal_overlay_path', '')
                
                print(f"  ✓ Extracted {num_features} features")
                print(f"  🌀 Fractal: {fractal_eq}")
                if overlay_path:
                    print(f"  🖼️  Overlay: {os.path.basename(overlay_path)}")
                successful += 1
                
        except Exception as e:
            print(f"  ✗ Error processing {img_path}: {e}")
            failed += 1
    
    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"✓ Successfully processed: {successful} images")
    if failed > 0:
        print(f"✗ Failed to process: {failed} images")
    print(f"✓ Results saved to: {CSV_PATH}")
    print(f"✓ Fractal overlays saved to: fractal_overlays/")
    print(f"✓ Dataset ready for research and machine learning!")
    print(f"🆕 NEW: CSV now includes fractal equations and overlay image references!")

def show_feature_info():
    """Display information about all the features being extracted"""
    print("\n=== COMPREHENSIVE TEXTURE FEATURE SET + FRACTAL FITTING ===")
    print(f"Total features extracted per image: {len(CSV_HEADERS) - 3}")
    print()
    
    feature_groups = {
        "Basic Statistical Features (8)": [
            "mean_intensity", "std_dev", "variance", "skewness", "kurtosis", 
            "range_intensity", "min_intensity", "max_intensity"
        ],
        "Entropy Measures (2)": [
            "entropy_shannon", "entropy_local"
        ],
        "Edge & Gradient Features (4)": [
            "edge_density", "edge_magnitude_mean", "edge_magnitude_std", "edge_orientation_std"
        ],
        "Haralick Texture Features (6)": [
            "haralick_contrast", "haralick_dissimilarity", "haralick_homogeneity", 
            "haralick_energy", "haralick_correlation", "haralick_asm"
        ],
        "Local Binary Pattern Features (3)": [
            "lbp_uniform_mean", "lbp_variance", "lbp_entropy"
        ],
        "Fractal Dimension Measures (5)": [
            "fractal_dim_higuchi", "fractal_dim_katz", "fractal_dim_dfa", 
            "fractal_dim_boxcount", "lacunarity"
        ],
        "Wavelet Features (5)": [
            "wavelet_energy_approx", "wavelet_energy_horizontal", "wavelet_energy_vertical", 
            "wavelet_energy_diagonal", "wavelet_entropy"
        ],
        "Tamura Texture Features (3)": [
            "tamura_coarseness", "tamura_contrast", "tamura_directionality"
        ],
        "Morphological Features (4)": [
            "area_coverage", "circularity", "solidity", "perimeter_complexity"
        ],
        "🆕 NEW: Fractal Surface Fitting (5)": [
            "fractal_equation", "fractal_hurst_exponent", "fractal_amplitude_scaling",
            "fractal_goodness_of_fit", "fractal_fitting_success"
        ],
        "🆕 NEW: Visualization Output (1)": [
            "fractal_overlay_path"
        ]
    }
    
    for group_name, features in feature_groups.items():
        print(f"{group_name}:")
        for feature in features:
            print(f"  • {feature}")
        print()
    
    print("🌀 FRACTAL SURFACE FITTING DETAILS:")
    print("  • Fits fractional Brownian motion (fBm) surface to each image")
    print("  • Extracts Hurst exponent (surface roughness parameter)")
    print("  • Generates mathematical equation: fBm(H=0.xxx, σ=x.xxx)")
    print("  • Creates overlay visualization showing fitted fractal")
    print("  • Saves overlay image for visual inspection")
    print()
    
    print("These features are ideal for:")
    print("  • Material classification and quality assessment")
    print("  • Fabric wear and degradation analysis") 
    print("  • Surface texture characterization")
    print("  • Machine learning model training")
    print("  • Research in material science and textile engineering")
    print("  🆕 • Fractal modeling and surface reconstruction")
    print("  🆕 • Visual validation of texture analysis results")
    print()

def main():
    """Main menu for the comprehensive fractal texture analysis system"""
    print("=== COMPREHENSIVE FRACTAL TEXTURE ANALYSIS SYSTEM ===")
    print("Advanced texture analysis with 45+ features per image")
    print("🆕 NEW: Now includes fractal surface fitting and overlay generation!")
    print()
    print("1. Capture new images and analyze them (Raspberry Pi mode)")
    print("2. Analyze existing images only")
    print("3. Show detailed feature information")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                if RASPBERRY_PI_MODE:
                    capture_and_analyze_workflow()
                else:
                    print("Error: Raspberry Pi camera not available. Use option 2 to analyze existing images.")
            elif choice == "2":
                analyze_existing_images()
            elif choice == "3":
                show_feature_info()
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
