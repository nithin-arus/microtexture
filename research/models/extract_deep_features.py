"""
Extract deep learning features from images.

Command-line script to extract ResNet50 and EfficientNetB0 features from images
and save to CSV for use in research pipeline.
"""

import argparse
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.models.deep_learning_models import DeepTextureExtractor
from research.utils.data_manifest import load_manifest, validate_labels
from research.utils.determinism import set_global_seeds


def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for deep feature extraction."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def extract_deep_features_for_manifest(manifest_path: str, images_dir: str, 
                                      output_csv: str, seed: int = 42):
    """
    Extract deep features from images listed in manifest.
    
    Parameters:
    -----------
    manifest_path : str
        Path to manifest CSV with filename, label columns
    images_dir : str
        Root directory containing images (searched recursively)
    output_csv : str
        Path to save deep_features.csv
    seed : int
        Random seed for deterministic behavior
    """
    # Set deterministic behavior
    set_global_seeds(seed)
    
    # Set TensorFlow/Keras deterministic flags
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Load manifest
    print(f"Loading manifest from {manifest_path}...")
    manifest_dict, class_list = load_manifest(manifest_path)
    print(f"Loaded {len(manifest_dict)} entries from manifest")
    
    # Initialize feature extractor
    print("Initializing deep feature extractors...")
    extractor = DeepTextureExtractor(input_size=(224, 224), include_top=False)
    
    # Get feature dimensions
    models_to_use = ['resnet50', 'efficientnet_b0']
    dims = {
        'resnet50': 2048,
        'efficientnet_b0': 1280
    }
    
    # Find all images
    images_dir_path = Path(images_dir)
    image_paths = []
    filenames = []
    labels = []
    
    print(f"Searching for images in {images_dir}...")
    for filename, label in manifest_dict.items():
        # Search for image file
        found = False
        for ext in ['.jpg', '.jpeg', '.png']:
            # Try direct path
            img_path = images_dir_path / filename
            if img_path.exists():
                image_paths.append(str(img_path))
                filenames.append(filename)
                labels.append(label)
                found = True
                break
            
            # Try in subdirectories
            for img_file in images_dir_path.rglob(f"**/{filename}"):
                if img_file.exists():
                    image_paths.append(str(img_file))
                    filenames.append(filename)
                    labels.append(label)
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            print(f"Warning: Image not found for {filename}")
    
    print(f"Found {len(image_paths)} images to process")
    
    # Extract features
    print("Extracting deep features...")
    all_features = []
    
    for i, (img_path, filename, label) in enumerate(zip(image_paths, filenames, labels)):
        try:
            # Load image
            img = load_image(img_path)
            
            # Extract features
            features_dict = extractor.extract_deep_features(img, models_to_use=models_to_use)
            
            # Create feature row
            row = {'filename': filename, 'label': label}
            
            # Add ResNet50 features
            if 'resnet50' in features_dict:
                resnet_features = features_dict['resnet50'].flatten()
                for j, feat in enumerate(resnet_features):
                    row[f'feat_resnet50_{j}'] = float(feat)
            else:
                # Fill with zeros if extraction failed
                for j in range(dims['resnet50']):
                    row[f'feat_resnet50_{j}'] = 0.0
            
            # Add EfficientNetB0 features
            if 'efficientnet_b0' in features_dict:
                efficientnet_features = features_dict['efficientnet_b0'].flatten()
                for j, feat in enumerate(efficientnet_features):
                    row[f'feat_efficientnet_{j}'] = float(feat)
            else:
                # Fill with zeros if extraction failed
                for j in range(dims['efficientnet_b0']):
                    row[f'feat_efficientnet_{j}'] = 0.0
            
            all_features.append(row)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(image_paths)} images...")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # Add row with zeros
            row = {'filename': filename, 'label': label}
            for j in range(dims['resnet50']):
                row[f'feat_resnet50_{j}'] = 0.0
            for j in range(dims['efficientnet_b0']):
                row[f'feat_efficientnet_{j}'] = 0.0
            all_features.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved deep features to {output_csv}")
    print(f"Feature matrix shape: {df.shape}")
    
    # Save metadata
    meta_path = output_path.parent / 'deep_features_meta.json'
    metadata = {
        'models': models_to_use,
        'dims': dims,
        'n_samples': len(df),
        'n_features': len(df.columns) - 2,  # Exclude filename and label
        'seed': seed
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract deep learning features from images')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to manifest CSV with filename,label columns')
    parser.add_argument('--images-dir', type=str, required=True,
                       help='Root directory containing images')
    parser.add_argument('--output-csv', type=str, required=True,
                       help='Path to save deep_features.csv')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for deterministic behavior')
    
    args = parser.parse_args()
    
    extract_deep_features_for_manifest(
        args.manifest,
        args.images_dir,
        args.output_csv,
        args.seed
    )


if __name__ == '__main__':
    main()

