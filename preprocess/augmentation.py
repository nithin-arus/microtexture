import numpy as np
import cv2
from scipy import ndimage
from scipy.ndimage import rotate, gaussian_filter, zoom
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import random
from typing import Tuple, List, Dict, Optional, Union
import warnings


class TextureAugmentation:
    """
    Comprehensive augmentation pipeline for texture analysis.
    
    Provides geometric, photometric, and multi-scale augmentations while
    preserving important texture properties for material classification.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the texture augmentation pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducible augmentations
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
    def geometric_augmentation(self, image: np.ndarray, 
                             apply_rotation: bool = True,
                             apply_flip: bool = True,
                             rotation_angles: List[float] = [90, 180, 270]) -> List[np.ndarray]:
        """
        Apply geometric augmentations that preserve texture properties.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        apply_rotation : bool
            Whether to apply rotations
        apply_flip : bool
            Whether to apply flips
        rotation_angles : List[float]
            Rotation angles to apply (in degrees)
            
        Returns:
        --------
        List[np.ndarray] : List of augmented images
        """
        augmented_images = [image.copy()]  # Include original
        
        if apply_rotation:
            for angle in rotation_angles:
                rotated = rotate(image, angle, reshape=False, mode='reflect')
                augmented_images.append(rotated)
                
        if apply_flip:
            # Horizontal flip
            h_flip = np.fliplr(image)
            augmented_images.append(h_flip)
            
            # Vertical flip
            v_flip = np.flipud(image)
            augmented_images.append(v_flip)
            
            # Both flips
            hv_flip = np.flipud(np.fliplr(image))
            augmented_images.append(hv_flip)
            
        return augmented_images
    
    def photometric_augmentation(self, image: np.ndarray,
                               brightness_range: Tuple[float, float] = (0.8, 1.2),
                               contrast_range: Tuple[float, float] = (0.8, 1.2),
                               gamma_range: Tuple[float, float] = (0.8, 1.2),
                               noise_std: float = 0.01) -> List[np.ndarray]:
        """
        Apply photometric augmentations to simulate different imaging conditions.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (assumed to be in [0, 1] range)
        brightness_range : Tuple[float, float]
            Range for brightness adjustment
        contrast_range : Tuple[float, float]
            Range for contrast adjustment
        gamma_range : Tuple[float, float]
            Range for gamma correction
        noise_std : float
            Standard deviation for Gaussian noise
            
        Returns:
        --------
        List[np.ndarray] : List of augmented images
        """
        augmented_images = []
        
        # Ensure image is in proper range
        if image.max() > 1.0:
            image = image / 255.0
            
        # Brightness adjustment
        brightness_factor = np.random.uniform(*brightness_range)
        bright_img = np.clip(image * brightness_factor, 0, 1)
        augmented_images.append(bright_img)
        
        # Contrast adjustment
        contrast_factor = np.random.uniform(*contrast_range)
        mean = np.mean(image)
        contrast_img = np.clip((image - mean) * contrast_factor + mean, 0, 1)
        augmented_images.append(contrast_img)
        
        # Gamma correction
        gamma = np.random.uniform(*gamma_range)
        gamma_img = np.clip(np.power(image, gamma), 0, 1)
        augmented_images.append(gamma_img)
        
        # Gaussian noise
        noise = np.random.normal(0, noise_std, image.shape)
        noisy_img = np.clip(image + noise, 0, 1)
        augmented_images.append(noisy_img)
        
        return augmented_images
    
    def multi_scale_augmentation(self, image: np.ndarray,
                               scale_factors: List[float] = [0.8, 1.2, 1.5],
                               crop_size: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """
        Apply multi-scale augmentations to capture texture at different resolutions.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        scale_factors : List[float]
            Scale factors for resizing
        crop_size : Tuple[int, int], optional
            Size for random cropping (height, width)
            
        Returns:
        --------
        List[np.ndarray] : List of augmented images
        """
        augmented_images = []
        original_shape = image.shape[:2]
        
        for scale in scale_factors:
            # Scale image
            if len(image.shape) == 3:
                scaled = zoom(image, (scale, scale, 1), mode='reflect')
            else:
                scaled = zoom(image, scale, mode='reflect')
                
            # Crop or pad to original size
            if scale > 1.0:
                # Crop from center if scaled up
                start_h = (scaled.shape[0] - original_shape[0]) // 2
                start_w = (scaled.shape[1] - original_shape[1]) // 2
                cropped = scaled[start_h:start_h + original_shape[0],
                               start_w:start_w + original_shape[1]]
                augmented_images.append(cropped)
            else:
                # Pad if scaled down
                pad_h = (original_shape[0] - scaled.shape[0]) // 2
                pad_w = (original_shape[1] - scaled.shape[1]) // 2
                
                if len(image.shape) == 3:
                    padded = np.pad(scaled, 
                                  ((pad_h, original_shape[0] - scaled.shape[0] - pad_h),
                                   (pad_w, original_shape[1] - scaled.shape[1] - pad_w),
                                   (0, 0)), mode='reflect')
                else:
                    padded = np.pad(scaled,
                                  ((pad_h, original_shape[0] - scaled.shape[0] - pad_h),
                                   (pad_w, original_shape[1] - scaled.shape[1] - pad_w)),
                                  mode='reflect')
                augmented_images.append(padded)
                
        # Random crops if crop_size specified
        if crop_size and crop_size[0] < original_shape[0] and crop_size[1] < original_shape[1]:
            for _ in range(3):  # Generate 3 random crops
                start_h = np.random.randint(0, original_shape[0] - crop_size[0] + 1)
                start_w = np.random.randint(0, original_shape[1] - crop_size[1] + 1)
                
                crop = image[start_h:start_h + crop_size[0],
                           start_w:start_w + crop_size[1]]
                
                # Resize back to original size
                if len(image.shape) == 3:
                    resized = zoom(crop, 
                                 (original_shape[0] / crop_size[0],
                                  original_shape[1] / crop_size[1], 1),
                                 mode='reflect')
                else:
                    resized = zoom(crop,
                                 (original_shape[0] / crop_size[0],
                                  original_shape[1] / crop_size[1]),
                                 mode='reflect')
                augmented_images.append(resized)
                
        return augmented_images
    
    def augment_image_comprehensive(self, image: np.ndarray,
                                  num_augmentations: int = 10) -> List[np.ndarray]:
        """
        Apply comprehensive augmentation combining multiple techniques.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
        num_augmentations : int
            Number of augmented versions to generate
            
        Returns:
        --------
        List[np.ndarray] : List of augmented images
        """
        all_augmented = []
        
        # Geometric augmentations
        geometric_augs = self.geometric_augmentation(image)
        all_augmented.extend(geometric_augs)
        
        # Photometric augmentations
        photometric_augs = self.photometric_augmentation(image)
        all_augmented.extend(photometric_augs)
        
        # Multi-scale augmentations
        multiscale_augs = self.multi_scale_augmentation(image)
        all_augmented.extend(multiscale_augs)
        
        # Randomly select from all augmentations
        if len(all_augmented) > num_augmentations:
            indices = np.random.choice(len(all_augmented), num_augmentations, replace=False)
            selected_augs = [all_augmented[i] for i in indices]
        else:
            selected_augs = all_augmented
            
        return selected_augs


class FeatureAugmentation(BaseEstimator, TransformerMixin):
    """
    Feature-level augmentation for texture feature vectors.
    
    Applies augmentation techniques directly in feature space to increase
    dataset diversity and improve model generalization.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize feature augmentation.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducible augmentations
        """
        self.random_state = random_state
        self.smote = SMOTE(random_state=random_state)
        
    def mixup_features(self, X: np.ndarray, y: np.ndarray, 
                      alpha: float = 0.2, 
                      num_mixup: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation to feature vectors.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        alpha : float
            Mixup interpolation parameter
        num_mixup : int, optional
            Number of mixup samples to generate
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : Augmented features and labels
        """
        if num_mixup is None:
            num_mixup = len(X) // 2
            
        mixup_X = []
        mixup_y = []
        
        for _ in range(num_mixup):
            # Sample two random indices
            idx1, idx2 = np.random.choice(len(X), 2, replace=False)
            
            # Sample mixup coefficient
            lam = np.random.beta(alpha, alpha)
            
            # Mix features
            mixed_x = lam * X[idx1] + (1 - lam) * X[idx2]
            
            # Mix labels (for regression) or keep dominant label (for classification)
            if len(np.unique(y)) <= 10:  # Assume classification
                mixed_y = y[idx1] if lam > 0.5 else y[idx2]
            else:  # Regression
                mixed_y = lam * y[idx1] + (1 - lam) * y[idx2]
                
            mixup_X.append(mixed_x)
            mixup_y.append(mixed_y)
            
        # Combine original and mixup data
        augmented_X = np.vstack([X, np.array(mixup_X)])
        augmented_y = np.concatenate([y, np.array(mixup_y)])
        
        return augmented_X, augmented_y
    
    def add_gaussian_noise(self, X: np.ndarray, 
                          noise_factor: float = 0.05) -> np.ndarray:
        """
        Add Gaussian noise to feature vectors.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        noise_factor : float
            Standard deviation of noise as fraction of feature std
            
        Returns:
        --------
        np.ndarray : Noisy features
        """
        # Calculate feature-wise standard deviations
        feature_stds = np.std(X, axis=0)
        
        # Generate noise
        noise = np.random.normal(0, noise_factor * feature_stds, X.shape)
        
        return X + noise
    
    def feature_dropout(self, X: np.ndarray, 
                       dropout_rate: float = 0.1) -> np.ndarray:
        """
        Apply feature dropout augmentation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        dropout_rate : float
            Fraction of features to zero out
            
        Returns:
        --------
        np.ndarray : Features with dropout applied
        """
        mask = np.random.binomial(1, 1 - dropout_rate, X.shape)
        return X * mask
    
    def synthetic_minority_oversampling(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE for handling class imbalance.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : Resampled features and labels
        """
        try:
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Warning: SMOTE failed ({e}), returning original data")
            return X, y
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for feature augmentation)."""
        return self
    
    def transform(self, X: np.ndarray, y: np.ndarray = None,
                 apply_mixup: bool = True,
                 apply_noise: bool = True,
                 apply_dropout: bool = False,
                 apply_smote: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Apply feature augmentation transformations.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray, optional
            Labels (required for mixup and SMOTE)
        apply_mixup : bool
            Whether to apply mixup augmentation
        apply_noise : bool
            Whether to add Gaussian noise
        apply_dropout : bool
            Whether to apply feature dropout
        apply_smote : bool
            Whether to apply SMOTE oversampling
            
        Returns:
        --------
        np.ndarray or Tuple[np.ndarray, np.ndarray] : Augmented features (and labels if provided)
        """
        X_aug = X.copy()
        y_aug = y.copy() if y is not None else None
        
        # Apply SMOTE first if requested
        if apply_smote and y_aug is not None:
            X_aug, y_aug = self.synthetic_minority_oversampling(X_aug, y_aug)
            
        # Apply mixup if requested
        if apply_mixup and y_aug is not None:
            X_aug, y_aug = self.mixup_features(X_aug, y_aug)
            
        # Apply noise
        if apply_noise:
            X_aug = self.add_gaussian_noise(X_aug)
            
        # Apply dropout
        if apply_dropout:
            X_aug = self.feature_dropout(X_aug)
            
        if y_aug is not None:
            return X_aug, y_aug
        else:
            return X_aug


class ImagingConditionSimulator:
    """
    Simulate different imaging conditions to improve model generalization.
    
    Creates variations in lighting, camera settings, and environmental conditions
    that might be encountered in real-world texture analysis scenarios.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the imaging condition simulator.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducible simulations
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def simulate_lighting_conditions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate different lighting conditions.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
            
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary of images under different lighting
        """
        lighting_variants = {}
        
        # Ensure image is in [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
            
        # Uniform low lighting
        low_light = np.clip(image * 0.3, 0, 1)
        lighting_variants['low_light'] = low_light
        
        # Uniform bright lighting
        bright_light = np.clip(image * 1.8, 0, 1)
        lighting_variants['bright_light'] = bright_light
        
        # Directional lighting (gradient)
        h, w = image.shape[:2]
        gradient = np.linspace(0.5, 1.5, w).reshape(1, -1)
        if len(image.shape) == 3:
            gradient = gradient[..., np.newaxis]
        directional = np.clip(image * gradient, 0, 1)
        lighting_variants['directional'] = directional
        
        # Uneven lighting (spot pattern)
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        spot_mask = 1.5 - (distance / max_distance)
        if len(image.shape) == 3:
            spot_mask = spot_mask[..., np.newaxis]
        spot_lighting = np.clip(image * spot_mask, 0, 1)
        lighting_variants['spot_lighting'] = spot_lighting
        
        return lighting_variants
    
    def simulate_camera_effects(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate different camera effects and settings.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
            
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary of images with different camera effects
        """
        camera_variants = {}
        
        # Motion blur
        kernel_size = 7
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0 / kernel_size
        motion_blur = cv2.filter2D(image, -1, kernel)
        camera_variants['motion_blur'] = motion_blur
        
        # Gaussian blur (out of focus)
        gaussian_blur = gaussian_filter(image, sigma=1.5)
        camera_variants['gaussian_blur'] = gaussian_blur
        
        # Depth of field simulation
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        blur_strength = (distance / max_distance) * 2.0
        
        dof_image = image.copy()
        for i in range(h):
            for j in range(w):
                sigma = blur_strength[i, j]
                if sigma > 0.1:
                    # Apply local blur
                    region = image[max(0, i-2):min(h, i+3), max(0, j-2):min(w, j+3)]
                    if region.size > 0:
                        blurred_region = gaussian_filter(region, sigma=min(sigma, 2.0))
                        dof_image[max(0, i-2):min(h, i+3), max(0, j-2):min(w, j+3)] = blurred_region
        
        camera_variants['depth_of_field'] = dof_image
        
        return camera_variants
    
    def simulate_resolution_changes(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate different resolution and compression effects.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
            
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary of images at different resolutions
        """
        resolution_variants = {}
        original_shape = image.shape[:2]
        
        # Low resolution
        low_res_scale = 0.5
        if len(image.shape) == 3:
            low_res = zoom(image, (low_res_scale, low_res_scale, 1), mode='reflect')
            upscaled = zoom(low_res, (1/low_res_scale, 1/low_res_scale, 1), mode='reflect')
        else:
            low_res = zoom(image, low_res_scale, mode='reflect')
            upscaled = zoom(low_res, 1/low_res_scale, mode='reflect')
            
        # Crop/pad to original size
        if upscaled.shape[0] != original_shape[0] or upscaled.shape[1] != original_shape[1]:
            if len(image.shape) == 3:
                upscaled_resized = np.zeros_like(image)
                min_h = min(upscaled.shape[0], original_shape[0])
                min_w = min(upscaled.shape[1], original_shape[1])
                upscaled_resized[:min_h, :min_w] = upscaled[:min_h, :min_w]
            else:
                upscaled_resized = np.zeros(original_shape)
                min_h = min(upscaled.shape[0], original_shape[0])
                min_w = min(upscaled.shape[1], original_shape[1])
                upscaled_resized[:min_h, :min_w] = upscaled[:min_h, :min_w]
            upscaled = upscaled_resized
            
        resolution_variants['low_resolution'] = upscaled
        
        # High resolution with downsampling
        high_res_scale = 2.0
        if len(image.shape) == 3:
            high_res = zoom(image, (high_res_scale, high_res_scale, 1), mode='reflect')
            downscaled = zoom(high_res, (1/high_res_scale, 1/high_res_scale, 1), mode='reflect')
        else:
            high_res = zoom(image, high_res_scale, mode='reflect')
            downscaled = zoom(high_res, 1/high_res_scale, mode='reflect')
            
        resolution_variants['high_resolution_downsampled'] = downscaled
        
        return resolution_variants
    
    def apply_comprehensive_simulation(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply comprehensive imaging condition simulation.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image
            
        Returns:
        --------
        Dict[str, np.ndarray] : Dictionary of all simulated conditions
        """
        all_variants = {'original': image}
        
        # Add lighting variations
        lighting_variants = self.simulate_lighting_conditions(image)
        all_variants.update(lighting_variants)
        
        # Add camera effects
        camera_variants = self.simulate_camera_effects(image)
        all_variants.update(camera_variants)
        
        # Add resolution changes
        resolution_variants = self.simulate_resolution_changes(image)
        all_variants.update(resolution_variants)
        
        return all_variants 