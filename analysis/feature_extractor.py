import cv2
import numpy as np
import pandas as pd
import os
from scipy.stats import entropy as scipy_entropy, skew, kurtosis
from skimage.filters import sobel, rank
from skimage.feature import canny, graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import disk, square
from skimage.measure import shannon_entropy
import pywt
import mahotas
from analysis.fractal_analysis import higuchi_fd, katz_fd, dfa_fd
from analysis.fractal_fitting import fit_and_visualize_fractal

def preprocess_image(img_path, resize_dim=(256, 256)):
    """Load and preprocess image for analysis"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.resize(img, resize_dim)
    return img

def compute_basic_stats(image):
    """Compute basic statistical measures"""
    pixels = image.flatten()
    return {
        'mean_intensity': np.mean(pixels),
        'std_dev': np.std(pixels),
        'skewness': skew(pixels),
        'kurtosis': kurtosis(pixels),
        'range_intensity': np.max(pixels) - np.min(pixels),
        'min_intensity': np.min(pixels),
        'max_intensity': np.max(pixels)
    }

def compute_entropy_features(image):
    """
    Compute various entropy measures.
    
    Local entropy computed via rank.entropy with 9x9 square structuring element.
    Histogram bins=256, base=2, boundary handling='constant'.
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_prob = hist / hist.sum()
    hist_prob = hist_prob[hist_prob > 0]  # Remove zeros to avoid log(0)
    
    # Compute local entropy using rank.entropy with 9x9 window
    try:
        # Convert to uint8 if needed
        if image.max() > 255:
            image_uint8 = (image / image.max() * 255).astype(np.uint8)
        elif image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
        
        # Compute local entropy map with 9x9 square structuring element
        selem = square(9)
        local_entropy_map = rank.entropy(image_uint8, selem)
        entropy_local = np.mean(local_entropy_map)
    except Exception as e:
        # Fallback to shannon entropy if rank.entropy fails
        entropy_local = shannon_entropy(image)
    
    return {
        'entropy_shannon': scipy_entropy(hist_prob, base=2),
        'entropy_local': entropy_local
    }

def compute_edge_features(image):
    """Compute edge and gradient-based features"""
    # Canny edge detection
    edges = canny(image / 255.0)
    edge_density = np.sum(edges) / edges.size
    
    # Sobel gradients
    grad_x = sobel(image, axis=1)
    grad_y = sobel(image, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Edge orientation
    grad_orientation = np.arctan2(grad_y, grad_x)
    grad_orientation = grad_orientation[grad_magnitude > np.mean(grad_magnitude)]
    
    return {
        'edge_density': edge_density,
        'edge_magnitude_mean': np.mean(grad_magnitude),
        'edge_magnitude_std': np.std(grad_magnitude),
        'edge_orientation_std': np.std(grad_orientation) if len(grad_orientation) > 0 else 0
    }

def compute_haralick_features(image):
    """Compute Haralick texture features using GLCM"""
    try:
        # Ensure image is in correct format and range
        if image.max() > 1:
            image_norm = (image / image.max() * 255).astype(np.uint8)
        else:
            image_norm = (image * 255).astype(np.uint8)
        
        # Create GLCM matrix with multiple angles
        distances = [1, 2]
        angles = [0, 45, 90, 135]
        glcm = graycomatrix(image_norm, distances=distances, angles=angles, 
                           levels=256, symmetric=True, normed=True)
        
        # Calculate Haralick properties (excluding ASM which is duplicate of energy)
        features = {}
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            features[f'haralick_{prop.lower()}'] = np.mean(values)
        
        return features
    except Exception as e:
        # Return default values if computation fails
        return {
            'haralick_contrast': 0.0,
            'haralick_dissimilarity': 0.0,
            'haralick_homogeneity': 0.0,
            'haralick_energy': 0.0,
            'haralick_correlation': 0.0
        }

def compute_lbp_features(image):
    """Compute Local Binary Pattern features"""
    try:
        radius = 1
        n_points = 8 * radius
        
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # Calculate LBP histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return {
            'lbp_uniform_mean': np.mean(hist[:-1]),  # Exclude non-uniform bin
            'lbp_variance': np.var(lbp),
            'lbp_entropy': scipy_entropy(hist[hist > 0], base=2)
        }
    except Exception as e:
        return {
            'lbp_uniform_mean': 0.0,
            'lbp_variance': 0.0,
            'lbp_entropy': 0.0
        }

def compute_fractal_features(image):
    """Compute all fractal dimension measures"""
    # Flatten the image for 1D-based FD methods
    signal = image.flatten().astype(np.float64)

    features = {}
    
    # Existing fractal methods
    try:
        features['fractal_dim_higuchi'] = higuchi_fd(signal)
    except:
        features['fractal_dim_higuchi'] = 1.0
        
    try:
        features['fractal_dim_katz'] = katz_fd(signal)
    except:
        features['fractal_dim_katz'] = 1.0
        
    try:
        features['fractal_dim_dfa'] = dfa_fd(signal)
    except:
        features['fractal_dim_dfa'] = 1.0
    
    # Box-counting fractal dimension
    try:
        features['fractal_dim_boxcount'] = compute_box_counting_dimension(image)
    except:
        features['fractal_dim_boxcount'] = 1.0
    
    # Lacunarity
    try:
        features['lacunarity'] = compute_lacunarity(image)
    except:
        features['lacunarity'] = 0.0
    
    return features

def compute_box_counting_dimension(image):
    """Compute box-counting fractal dimension"""
    # Convert to binary
    threshold = np.mean(image)
    binary_img = (image > threshold).astype(np.uint8)
    
    # Find the maximum box size
    max_box_size = min(binary_img.shape) // 2
    box_sizes = [2**i for i in range(1, int(np.log2(max_box_size)) + 1)]
    
    counts = []
    for box_size in box_sizes:
        count = 0
        for i in range(0, binary_img.shape[0], box_size):
            for j in range(0, binary_img.shape[1], box_size):
                box = binary_img[i:i+box_size, j:j+box_size]
                if np.any(box):
                    count += 1
        counts.append(count)
    
    if len(counts) < 2:
        return 1.0
    
    # Linear regression on log-log plot
    log_box_sizes = np.log(box_sizes)
    log_counts = np.log(counts)
    
    coeffs = np.polyfit(log_box_sizes, log_counts, 1)
    return -coeffs[0]  # Negative slope is the fractal dimension

def compute_lacunarity(image, box_sizes=None):
    """Compute lacunarity measure"""
    if box_sizes is None:
        max_size = min(image.shape) // 4
        box_sizes = range(2, max_size, 2)
    
    lacunarity_values = []
    
    for box_size in box_sizes:
        mass_values = []
        
        for i in range(0, image.shape[0] - box_size + 1, box_size//2):
            for j in range(0, image.shape[1] - box_size + 1, box_size//2):
                box = image[i:i+box_size, j:j+box_size]
                mass = np.sum(box)
                mass_values.append(mass)
        
        if len(mass_values) > 1:
            mean_mass = np.mean(mass_values)
            var_mass = np.var(mass_values)
            if mean_mass > 0:
                lacunarity = (var_mass / (mean_mass**2)) + 1
                lacunarity_values.append(lacunarity)
    
    return np.mean(lacunarity_values) if lacunarity_values else 0.0

def compute_wavelet_features(image):
    """Compute wavelet-based features"""
    try:
        # Perform 2D wavelet decomposition
        coeffs2 = pywt.dwt2(image, 'db4')
        cA, (cH, cV, cD) = coeffs2
        
        # Energy in each subband
        energy_approx = np.sum(cA**2)
        energy_horizontal = np.sum(cH**2) 
        energy_vertical = np.sum(cV**2)
        energy_diagonal = np.sum(cD**2)
        
        total_energy = energy_approx + energy_horizontal + energy_vertical + energy_diagonal
        
        if total_energy > 0:
            # Normalize energies
            energy_approx /= total_energy
            energy_horizontal /= total_energy
            energy_vertical /= total_energy
            energy_diagonal /= total_energy
        
        # Wavelet entropy
        all_coeffs = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
        wavelet_entropy = scipy_entropy(np.abs(all_coeffs) + 1e-10, base=2)
        
        return {
            'wavelet_energy_approx': energy_approx,
            'wavelet_energy_horizontal': energy_horizontal,
            'wavelet_energy_vertical': energy_vertical,
            'wavelet_energy_diagonal': energy_diagonal,
            'wavelet_entropy': wavelet_entropy
        }
    except Exception as e:
        return {
            'wavelet_energy_approx': 0.0,
            'wavelet_energy_horizontal': 0.0,
            'wavelet_energy_vertical': 0.0,
            'wavelet_energy_diagonal': 0.0,
            'wavelet_entropy': 0.0
        }

def compute_tamura_features(image):
    """Compute Tamura texture features"""
    try:
        # Coarseness
        coarseness = compute_coarseness(image)
        
        # Contrast (Tamura definition)
        contrast_tamura = compute_contrast_tamura(image)
        
        # Directionality
        directionality = compute_directionality(image)
        
        return {
            'tamura_coarseness': coarseness,
            'tamura_contrast': contrast_tamura,
            'tamura_directionality': directionality
        }
    except Exception as e:
        return {
            'tamura_coarseness': 0.0,
            'tamura_contrast': 0.0,
            'tamura_directionality': 0.0
        }

def compute_coarseness(image):
    """Compute Tamura coarseness"""
    # Simple implementation of coarseness
    # Average of local variations at different scales
    scales = [1, 2, 4, 8]
    coarseness_values = []
    
    for scale in scales:
        # Average over neighborhoods
        kernel = np.ones((scale, scale)) / (scale * scale)
        smoothed = cv2.filter2D(image.astype(np.float32), -1, kernel)
        variation = np.var(smoothed)
        coarseness_values.append(variation)
    
    return np.mean(coarseness_values)

def compute_contrast_tamura(image):
    """Compute Tamura contrast"""
    return np.std(image) / (np.mean(image) + 1e-10)

def compute_directionality(image):
    """Compute Tamura directionality"""
    # Gradient-based directionality
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    angles = np.arctan2(grad_y, grad_x)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Weight by gradient magnitude
    weighted_angles = angles[magnitude > np.mean(magnitude)]
    
    if len(weighted_angles) > 0:
        return np.std(weighted_angles)
    else:
        return 0.0

def compute_morphological_features(image):
    """Compute morphological features"""
    try:
        # Binary version for morphological analysis
        threshold = np.mean(image)
        binary = (image > threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            # Features
            area_coverage = area / (image.shape[0] * image.shape[1])
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0
            
            return {
                'area_coverage': area_coverage,
                'circularity': circularity,
                'solidity': solidity,
                'perimeter_complexity': perimeter / np.sqrt(area) if area > 0 else 0
            }
        else:
            return {
                'area_coverage': 0.0,
                'circularity': 0.0,
                'solidity': 0.0,
                'perimeter_complexity': 0.0
            }
    except Exception as e:
        return {
            'area_coverage': 0.0,
            'circularity': 0.0,
            'solidity': 0.0,
            'perimeter_complexity': 0.0
        }

def extract_features(img_path, label=None, enable_fractal_fitting=True):
    """Extract comprehensive texture features from image including fractal fitting"""
    try:
        # Load and preprocess image
        img = preprocess_image(img_path)
        
        # Initialize features dictionary
        features = {
            "filename": os.path.basename(img_path),
            "path": img_path,
            "label": label
        }
        
        # Compute all feature groups
        features.update(compute_basic_stats(img))
        features.update(compute_entropy_features(img))
        features.update(compute_edge_features(img))
        features.update(compute_haralick_features(img))
        features.update(compute_lbp_features(img))
        features.update(compute_fractal_features(img))
        features.update(compute_wavelet_features(img))
        features.update(compute_tamura_features(img))
        features.update(compute_morphological_features(img))
        
        # Add fractal surface fitting if enabled
        if enable_fractal_fitting:
            try:
                print(f"  → Fitting fractal surface...")
                fractal_params, overlay_path = fit_and_visualize_fractal(img, img_path)
                features.update(fractal_params)
                print(f"  → Fractal fitting complete: {fractal_params.get('fractal_equation', 'N/A')}")
            except Exception as e:
                print(f"  → Fractal fitting failed: {e}")
                # Add default fractal parameters
                features.update({
                    'fractal_equation': 'fitting_failed',
                    'fractal_hurst_exponent': 0.0,
                    'fractal_amplitude_scaling': 0.0,
                    'fractal_spectrum_corr': 0.0,
                    'fractal_spectrum_rmse': 0.0,
                    'fractal_fitting_success': False,
                    'fractal_overlay_path': ''
                })
        
        # Round numerical values for cleaner output
        for key, value in features.items():
            if isinstance(value, (int, float)) and key not in ["filename", "path", "label", "fractal_equation", "fractal_overlay_path"]:
                features[key] = round(float(value), 6)
                
        return features
        
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        # Return minimal feature set in case of error
        return {
            "filename": os.path.basename(img_path) if img_path else "error",
            "path": img_path or "error",
            "label": label,
            "error": str(e)
        }

def batch_extract_features(img_dir, label_map=None, enable_fractal_fitting=True):
    """Extract features from multiple images"""
    results = []
    for fname in os.listdir(img_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            fpath = os.path.join(img_dir, fname)
            label = label_map[fname] if label_map and fname in label_map else None
            features = extract_features(fpath, label, enable_fractal_fitting)
            results.append(features)
    return pd.DataFrame(results)

def get_feature_schema():
    """
    Return ordered list of handcrafted feature names.
    
    Returns:
    --------
    list : Ordered feature names (38 handcrafted features)
        - 7 statistical: mean_intensity, std_dev, skewness, kurtosis, range_intensity, min_intensity, max_intensity
        - 2 entropy: entropy_shannon, entropy_local
        - 4 edge: edge_density, edge_magnitude_mean, edge_magnitude_std, edge_orientation_std
        - 5 haralick: haralick_contrast, haralick_dissimilarity, haralick_homogeneity, haralick_energy, haralick_correlation
        - 3 LBP: lbp_uniform_mean, lbp_variance, lbp_entropy
        - 5 fractal: fractal_dim_higuchi, fractal_dim_katz, fractal_dim_dfa, fractal_dim_boxcount, lacunarity
        - 5 wavelet: wavelet_energy_approx, wavelet_energy_horizontal, wavelet_energy_vertical, wavelet_energy_diagonal, wavelet_entropy
        - 3 Tamura: tamura_coarseness, tamura_contrast, tamura_directionality
        - 4 morphological: area_coverage, circularity, solidity, perimeter_complexity
    """
    return [
        # Statistical features (7)
        'mean_intensity', 'std_dev', 'skewness', 'kurtosis', 
        'range_intensity', 'min_intensity', 'max_intensity',
        # Entropy measures (2)
        'entropy_shannon', 'entropy_local',
        # Edge features (4)
        'edge_density', 'edge_magnitude_mean', 'edge_magnitude_std', 'edge_orientation_std',
        # Haralick features (5) - ASM removed (duplicate of energy)
        'haralick_contrast', 'haralick_dissimilarity', 'haralick_homogeneity', 
        'haralick_energy', 'haralick_correlation',
        # LBP features (3)
        'lbp_uniform_mean', 'lbp_variance', 'lbp_entropy',
        # Fractal dimension measures (5)
        'fractal_dim_higuchi', 'fractal_dim_katz', 'fractal_dim_dfa', 
        'fractal_dim_boxcount', 'lacunarity',
        # Wavelet features (5)
        'wavelet_energy_approx', 'wavelet_energy_horizontal', 'wavelet_energy_vertical', 
        'wavelet_energy_diagonal', 'wavelet_entropy',
        # Tamura features (3)
        'tamura_coarseness', 'tamura_contrast', 'tamura_directionality',
        # Morphological features (4)
        'area_coverage', 'circularity', 'solidity', 'perimeter_complexity'
    ]
