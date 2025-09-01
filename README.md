# Comprehensive Fractal Texture Analysis for Material Research

This project provides a complete pipeline for capturing images using a Raspberry Pi camera and extracting **40+ comprehensive texture features** for advanced material texture analysis. It's designed for research applications studying material wear, surface changes, texture classification, and machine learning dataset creation.

## 🎯 Features Extracted (40+ per image)

The system extracts comprehensive texture features organized into 9 categories:

### 📊 **Basic Statistical Features (8)**
- `mean_intensity`, `std_dev`, `variance`, `skewness`, `kurtosis`
- `range_intensity`, `min_intensity`, `max_intensity`

### 🔢 **Entropy Measures (2)**
- `entropy_shannon`: Shannon entropy of intensity histogram
- `entropy_local`: Local entropy measure

### 📐 **Edge & Gradient Features (4)**
- `edge_density`: Percentage of edge pixels (Canny detector)
- `edge_magnitude_mean`, `edge_magnitude_std`: Gradient statistics
- `edge_orientation_std`: Edge direction variability

### 🎨 **Haralick Texture Features (6)**
- `haralick_contrast`, `haralick_dissimilarity`, `haralick_homogeneity`
- `haralick_energy`, `haralick_correlation`, `haralick_asm`

### 🔍 **Local Binary Pattern Features (3)**
- `lbp_uniform_mean`: Uniform pattern distribution
- `lbp_variance`: LBP texture contrast
- `lbp_entropy`: LBP pattern randomness

### 🌀 **Fractal Dimension Measures (5)**
- `fractal_dim_higuchi`: Higuchi Fractal Dimension (complexity/roughness)
- `fractal_dim_katz`: Katz Fractal Dimension (alternative complexity)
- `fractal_dim_dfa`: Detrended Fluctuation Analysis (self-similarity)
- `fractal_dim_boxcount`: Box-counting fractal dimension
- `lacunarity`: Texture "gappiness" measure

### 🌊 **Wavelet Features (5)**
- `wavelet_energy_approx`, `wavelet_energy_horizontal`
- `wavelet_energy_vertical`, `wavelet_energy_diagonal`
- `wavelet_entropy`: Multi-scale texture analysis

### 📏 **Tamura Texture Features (3)**
- `tamura_coarseness`: Grain size of texture
- `tamura_contrast`: Dynamic range measure
- `tamura_directionality`: Preferred orientation strength

### 🔬 **Morphological Features (4)**
- `area_coverage`: Material vs background ratio
- `circularity`: Shape regularity measure
- `solidity`: Convexity measure
- `perimeter_complexity`: Boundary complexity

## Setup

### Prerequisites

- Python 3.8+
- Raspberry Pi with camera module (for image capture)
- Virtual environment (recommended)

### Installation

1. **Clone/Download the project**
2. **Set up virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Mac/Linux
   # or: venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Raspberry Pi Setup

If running on Raspberry Pi for image capture:
```bash
sudo apt update
sudo apt install python3-picamera2
```

## Usage

### Method 1: Complete Workflow (Recommended)

Run the main script for a complete capture and analysis workflow:

```bash
python main.py
```

This provides an interactive menu:
1. **Capture & Analyze**: Take new pictures and automatically extract ALL features
2. **Analyze Existing**: Process images already in the `capture/images/` folder
3. **Show Feature Info**: Display detailed information about all 40+ features
4. **Exit**: Close the program

### Method 2: Test Analysis Only

Test the comprehensive feature extraction on existing images:

```bash
python test_analysis.py [optional_image_path]
```

### Method 3: Individual Components

**Capture images only** (Raspberry Pi):
```bash
python capture/capture_image.py
```

**Analyze specific image**:
```bash
python -c "from analysis.feature_extractor import extract_features; print(extract_features('path/to/image.jpg', 'material_label'))"
```

## Project Structure

```
micro_texture_project/
├── main.py                    # Main workflow script (40+ features)
├── test_analysis.py           # Test script for comprehensive analysis
├── requirements.txt           # Python dependencies (including new ones)
├── README.md                  # This file
├── analysis/
│   ├── feature_extractor.py   # Comprehensive feature extraction (40+ features)
│   └── fractal_analysis.py    # Fractal dimension calculations
├── capture/
│   ├── capture_image.py       # Camera capture functions
│   └── images/                # Captured images (organized by label)
├── data/
│   └── features.csv           # Output CSV with ALL extracted features
├── features/                  # Legacy feature extraction (backup)
└── preprocess/                # Image preprocessing utilities
```

## Output

All extracted features are saved to `data/features.csv` with **43 columns**:

### Identification Columns
- `filename`: Image filename
- `path`: Full path to image
- `label`: Material label (e.g., 'cotton', 'denim')

### Feature Columns (40 total)
- **8 Basic Statistics**: mean_intensity, std_dev, variance, skewness, kurtosis, range_intensity, min_intensity, max_intensity
- **2 Entropy Measures**: entropy_shannon, entropy_local
- **4 Edge Features**: edge_density, edge_magnitude_mean, edge_magnitude_std, edge_orientation_std
- **6 Haralick Features**: haralick_contrast, haralick_dissimilarity, haralick_homogeneity, haralick_energy, haralick_correlation, haralick_asm
- **3 LBP Features**: lbp_uniform_mean, lbp_variance, lbp_entropy
- **5 Fractal Features**: fractal_dim_higuchi, fractal_dim_katz, fractal_dim_dfa, fractal_dim_boxcount, lacunarity
- **5 Wavelet Features**: wavelet_energy_approx, wavelet_energy_horizontal, wavelet_energy_vertical, wavelet_energy_diagonal, wavelet_entropy
- **3 Tamura Features**: tamura_coarseness, tamura_contrast, tamura_directionality
- **4 Morphological Features**: area_coverage, circularity, solidity, perimeter_complexity

## Workflow for Material Research

1. **Prepare samples**: Position material samples consistently under camera
2. **Capture baseline**: Take initial images of pristine material
3. **Apply wear/treatment**: Subject material to wear, washing, etc.
4. **Capture follow-up**: Take images at regular intervals
5. **Analyze changes**: Compare 40+ features across time/conditions
6. **Machine Learning**: Use comprehensive feature set for classification/prediction

## Research Applications

### 🧵 **Textile & Material Science**
- **Fabric wear analysis**: Track how materials change over wash cycles
- **Quality control**: Detect manufacturing inconsistencies
- **Material classification**: Distinguish between different fabric types
- **Surface defect detection**: Identify anomalies automatically

### 🤖 **Machine Learning & AI**
- **Feature-rich datasets**: 40+ features per image for ML training
- **Material property prediction**: Predict wear, durability, quality
- **Automated inspection**: Train models for quality assessment
- **Texture-based classification**: Distinguish materials by surface properties

### 🔬 **Advanced Research**
- **Multi-scale analysis**: Wavelet and fractal features capture different scales
- **Comparative studies**: Statistical framework for material comparison
- **Longitudinal studies**: Track material changes over time
- **Surface characterization**: Comprehensive texture profiling

## Example Research Workflow

```bash
# 1. Capture baseline cotton samples
python main.py  # Select option 1, label: "cotton_baseline"

# 2. Apply 10 wash cycles to samples

# 3. Capture after washing
python main.py  # Select option 1, label: "cotton_10washes"

# 4. Compare results in CSV
# The 40+ features allow detailed comparison of texture changes
```

## Performance & Timing

- **Feature extraction time**: ~5-15 seconds per image (depending on image size)
- **Memory usage**: Optimized for Raspberry Pi hardware
- **Robustness**: Error handling ensures analysis continues even if some features fail
- **Scalability**: Batch processing for large image collections

## Tips for Best Results

- **Consistent lighting**: Use uniform illumination across all captures
- **Fixed distance**: Keep camera at same distance from samples
- **Material positioning**: Ensure samples are flat and properly aligned
- **Multiple samples**: Capture 5+ images per condition for statistical significance
- **Label consistency**: Use clear, consistent material labels
- **Resolution**: Higher resolution images provide more detailed texture analysis

## Troubleshooting

**Import errors on Raspberry Pi**:
```bash
sudo apt install python3-picamera2
pip install mahotas PyWavelets
```

**Analysis works but no camera**:
- Use option 2 (Analyze existing images) in main menu
- Run `test_analysis.py` to verify feature extraction

**Empty CSV file**:
- Check that images exist in `capture/images/`
- Verify image file extensions (.jpg, .jpeg, .png)
- Run `test_analysis.py` to test individual image processing

**Feature extraction errors**:
- System includes robust error handling
- Failed features default to 0.0 while continuing extraction
- Check image quality and format

## Dependencies

The system uses these key libraries:
- **OpenCV**: Image processing and computer vision
- **scikit-image**: Advanced image analysis algorithms
- **PyWavelets**: Wavelet decomposition analysis
- **scipy**: Statistical analysis and signal processing
- **pandas**: Data manipulation and CSV output
- **numpy**: Numerical computations
- **mahotas**: Additional computer vision algorithms

## Research Citation

If you use this comprehensive texture analysis system in research, please cite the relevant algorithms:
- Higuchi, T. (1988). Approach to an irregular time series on the basis of the fractal theory
- Katz, M. J. (1988). Fractals and the analysis of waveforms  
- Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides
- Haralick, R. M., et al. (1973). Textural features for image classification
- Ojala, T., et al. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns

## License

This project is intended for academic and research use. The comprehensive feature set makes it particularly valuable for material science, textile engineering, and computer vision research. 