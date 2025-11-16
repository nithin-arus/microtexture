import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import optimize
from scipy.interpolate import griddata
import os
from datetime import datetime

class FractalSurfaceFitter:
    """
    Fits fractal surfaces to images and generates visualization overlays
    """
    
    def __init__(self):
        self.fitted_params = {}
        self.fractal_surface = None
        
    def fractional_brownian_motion_2d(self, shape, hurst=0.7, sigma=1.0, random_seed=None):
        """
        Generate 2D fractional Brownian motion surface
        
        Parameters:
        -----------
        shape : tuple
            (height, width) of the surface
        hurst : float
            Hurst exponent (0 < H < 1)
            H = 0.5: regular Brownian motion
            H > 0.5: persistent (smooth)
            H < 0.5: anti-persistent (rough)
        sigma : float
            Scaling factor for amplitude
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        h, w = shape
        
        # Create frequency grid
        x_freq = np.fft.fftfreq(w, d=1.0)
        y_freq = np.fft.fftfreq(h, d=1.0)
        fx, fy = np.meshgrid(x_freq, y_freq)
        
        # Avoid division by zero at DC component
        f_mag = np.sqrt(fx**2 + fy**2)
        f_mag[0, 0] = 1.0  # Set DC to 1 to avoid division by zero
        
        # Generate power spectrum for fractional Brownian motion
        # Power spectrum ~ 1/f^(2H+1) for 2D fBm
        power_spectrum = 1.0 / (f_mag**(2*hurst + 1))
        power_spectrum[0, 0] = 0  # Set DC component to 0
        
        # Generate random phases
        random_phase = 2 * np.pi * np.random.random((h, w))
        
        # Create complex spectrum
        amplitude = np.sqrt(power_spectrum * sigma)
        complex_spectrum = amplitude * np.exp(1j * random_phase)
        
        # Ensure Hermitian symmetry for real output
        complex_spectrum = self._ensure_hermitian_symmetry(complex_spectrum)
        
        # Inverse FFT to get the surface
        surface = np.real(np.fft.ifft2(complex_spectrum))
        
        return surface
    
    def _ensure_hermitian_symmetry(self, spectrum):
        """Ensure the spectrum has Hermitian symmetry for real ifft output"""
        h, w = spectrum.shape
        
        # Make sure DC and Nyquist are real
        spectrum[0, 0] = np.real(spectrum[0, 0])
        if w % 2 == 0:
            spectrum[0, w//2] = np.real(spectrum[0, w//2])
        if h % 2 == 0:
            spectrum[h//2, 0] = np.real(spectrum[h//2, 0])
            if w % 2 == 0:
                spectrum[h//2, w//2] = np.real(spectrum[h//2, w//2])
        
        return spectrum
    
    def fit_fractal_surface(self, image, method='fBm'):
        """
        Fit a fractal surface model to the image
        
        Parameters:
        -----------
        image : np.ndarray
            Grayscale image to fit
        method : str
            Fractal fitting method ('fBm', 'multifractal')
            
        Returns:
        --------
        dict : Fitted parameters and goodness of fit
        """
        if method == 'fBm':
            return self._fit_fbm_surface(image)
        else:
            raise NotImplementedError(f"Method {method} not implemented")
    
    def _fit_fbm_surface(self, image):
        """
        Fit fractional Brownian motion parameters to image using radial power spectrum matching.
        
        This method estimates the Hurst exponent (H) and amplitude scaling (σ) by:
        1. Computing the radial average of the image's power spectrum
        2. Generating synthetic fBm surfaces with candidate parameters
        3. Optimizing the Pearson correlation between image and synthetic radial power spectra
        
        Parameters:
        -----------
        image : np.ndarray
            Grayscale image to fit
            
        Returns:
        --------
        dict : Fitted parameters including:
            - hurst_exponent: Hurst exponent H ∈ [0.1, 0.9]
            - amplitude_scaling: Amplitude scaling σ ∈ [0.1, 100.0]
            - fractal_spectrum_corr: Pearson correlation [-1, 1] between radial power spectra
            - fractal_spectrum_rmse: RMSE ≥ 0 between radial power spectra
            - fractal_equation: String representation of fitted fBm
        
        Optimization:
        ------------
        - Method: L-BFGS-B (bounded optimization)
        - Bounds: H ∈ [0.1, 0.9], σ ∈ [0.1, 100.0]
        - Initial guess: H=0.5, σ=1.0
        - Maximum iterations: 50
        
        Note:
        ----
        The correlation coefficient is Pearson correlation, not R².
        Correlation ∈ [-1, 1] indicates similarity of power spectrum shapes.
        """
        # Normalize image to zero mean
        image_normalized = image.astype(np.float64)
        image_normalized = image_normalized - np.mean(image_normalized)
        
        # Compute image power spectrum (for final RMSE calculation)
        image_fft = np.fft.fft2(image_normalized)
        image_power = np.abs(image_fft)**2
        image_radial = self._radial_average(image_power)
        
        # Define optimization function
        def objective_function(params):
            hurst, sigma = params
            
            # Constrain parameters to valid ranges
            hurst = np.clip(hurst, 0.1, 0.9)
            sigma = np.clip(sigma, 0.1, 100.0)
            
            try:
                # Generate synthetic fractal surface
                synthetic_surface = self.fractional_brownian_motion_2d(
                    image.shape, hurst=hurst, sigma=sigma, random_seed=42
                )
                
                # Normalize synthetic surface
                synthetic_surface = synthetic_surface - np.mean(synthetic_surface)
                
                # Compute similarity metrics
                # 1. Correlation between power spectra
                synth_fft = np.fft.fft2(synthetic_surface)
                synth_power = np.abs(synth_fft)**2
                
                # Radial average of power spectra
                synth_radial = self._radial_average(synth_power)
                
                # Correlation coefficient between radial power spectra
                # Use log scale for better matching of power law behavior
                log_image_radial = np.log(image_radial + 1e-10)
                log_synth_radial = np.log(synth_radial + 1e-10)
                correlation = np.corrcoef(log_image_radial, log_synth_radial)[0, 1]
                
                # Return negative correlation (for minimization)
                return -correlation if not np.isnan(correlation) else 1.0
                
            except Exception as e:
                return 1.0  # Return bad fit if error occurs
        
        # Initial guess and bounds
        initial_guess = [0.5, 1.0]  # [hurst, sigma]
        bounds = [(0.1, 0.9), (0.1, 100.0)]  # H ∈ [0.1, 0.9], σ ∈ [0.1, 100.0]
        
        try:
            # Optimize parameters
            result = optimize.minimize(
                objective_function, 
                initial_guess, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            
            fitted_hurst, fitted_sigma = result.x
            spectrum_corr = -result.fun  # Convert back to positive correlation
            
            # Generate the best-fit fractal surface
            self.fractal_surface = self.fractional_brownian_motion_2d(
                image.shape, hurst=fitted_hurst, sigma=fitted_sigma, random_seed=42
            )
            
            # Compute RMSE between radial power spectra
            synth_fft = np.fft.fft2(self.fractal_surface - np.mean(self.fractal_surface))
            synth_power = np.abs(synth_fft)**2
            synth_radial = self._radial_average(synth_power)
            
            # Normalize for RMSE computation
            log_image_radial = np.log(image_radial + 1e-10)
            log_synth_radial = np.log(synth_radial + 1e-10)
            spectrum_rmse = np.sqrt(np.mean((log_image_radial - log_synth_radial)**2))
            
            # Store fitted parameters
            self.fitted_params = {
                'fractal_type': 'fractional_brownian_motion',
                'hurst_exponent': round(fitted_hurst, 6),
                'amplitude_scaling': round(fitted_sigma, 6),
                'fractal_spectrum_corr': round(spectrum_corr, 6),
                'fractal_spectrum_rmse': round(spectrum_rmse, 6),
                'fractal_equation': f'fBm(H={fitted_hurst:.3f}, σ={fitted_sigma:.3f})',
                'fitting_success': result.success
            }
            
            return self.fitted_params
            
        except Exception as e:
            # Return default parameters if fitting fails
            self.fitted_params = {
                'fractal_type': 'fractional_brownian_motion',
                'hurst_exponent': 0.5,
                'amplitude_scaling': 1.0,
                'fractal_spectrum_corr': 0.0,
                'fractal_spectrum_rmse': 0.0,
                'fractal_equation': 'fBm(H=0.500, σ=1.000)',
                'fitting_success': False,
                'error': str(e)
            }
            
            # Generate default fractal surface
            self.fractal_surface = self.fractional_brownian_motion_2d(
                image.shape, hurst=0.5, sigma=1.0, random_seed=42
            )
            
            return self.fitted_params
    
    def _radial_average(self, power_spectrum):
        """
        Compute radial average of 2D power spectrum
        """
        h, w = power_spectrum.shape
        center_x, center_y = w // 2, h // 2
        
        # Create coordinate grids
        x = np.arange(w) - center_x
        y = np.arange(h) - center_y
        X, Y = np.meshgrid(x, y)
        
        # Compute radial distances
        R = np.sqrt(X**2 + Y**2)
        
        # Define radial bins
        r_max = min(center_x, center_y)
        r_bins = np.linspace(0, r_max, min(r_max, 50))
        
        # Compute radial average
        radial_profile = []
        for i in range(len(r_bins) - 1):
            mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
            if np.any(mask):
                radial_profile.append(np.mean(power_spectrum[mask]))
            else:
                radial_profile.append(0)
        
        return np.array(radial_profile)
    
    def create_fractal_overlay(self, original_image, output_path, alpha=0.4, colormap='hot'):
        """
        Create visualization overlay of fitted fractal on original image
        
        Parameters:
        -----------
        original_image : np.ndarray
            Original grayscale image
        output_path : str
            Path to save the overlay image
        alpha : float
            Transparency of fractal overlay (0-1)
        colormap : str
            Matplotlib colormap for fractal visualization
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Normalize images for visualization
            original_norm = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            if self.fractal_surface is not None:
                fractal_norm = cv2.normalize(self.fractal_surface, None, 0, 255, cv2.NORM_MINMAX)
            else:
                # Generate default fractal if none exists
                fractal_norm = cv2.normalize(
                    self.fractional_brownian_motion_2d(original_image.shape, hurst=0.5, sigma=1.0),
                    None, 0, 255, cv2.NORM_MINMAX
                )
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_norm, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Fitted fractal surface
            im1 = axes[1].imshow(fractal_norm, cmap=colormap)
            axes[1].set_title(f'Fitted Fractal Surface\n{self.fitted_params.get("fractal_equation", "N/A")}')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], shrink=0.8)
            
            # Overlay
            axes[2].imshow(original_norm, cmap='gray')
            overlay = axes[2].imshow(fractal_norm, cmap=colormap, alpha=alpha)
            axes[2].set_title(f'Fractal Overlay (α={alpha})\nSpectrum Corr: {self.fitted_params.get("fractal_spectrum_corr", 0):.3f}')
            axes[2].axis('off')
            
            # Add fractal parameters as text
            param_text = f"""Fractal Parameters:
• Hurst Exponent: {self.fitted_params.get('hurst_exponent', 'N/A')}
• Amplitude Scaling: {self.fitted_params.get('amplitude_scaling', 'N/A')}
• Equation: {self.fitted_params.get('fractal_equation', 'N/A')}
• Spectrum Corr: {self.fitted_params.get('fractal_spectrum_corr', 0):.4f}
• Spectrum RMSE: {self.fitted_params.get('fractal_spectrum_rmse', 0):.4f}"""
            
            plt.figtext(0.02, 0.02, param_text, fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error creating fractal overlay: {e}")
            return False
    
    def get_fractal_parameters_for_csv(self):
        """
        Return fractal parameters formatted for CSV inclusion
        
        Returns:
        --------
        dict : Fractal parameters with keys:
            - fractal_equation: String representation
            - fractal_hurst_exponent: Hurst exponent H
            - fractal_amplitude_scaling: Amplitude scaling σ
            - fractal_spectrum_corr: Pearson correlation [-1, 1] (not R²)
            - fractal_spectrum_rmse: RMSE ≥ 0
            - fractal_fitting_success: Boolean indicating success
        """
        return {
            'fractal_equation': self.fitted_params.get('fractal_equation', ''),
            'fractal_hurst_exponent': self.fitted_params.get('hurst_exponent', 0.0),
            'fractal_amplitude_scaling': self.fitted_params.get('amplitude_scaling', 0.0),
            'fractal_spectrum_corr': self.fitted_params.get('fractal_spectrum_corr', 0.0),
            'fractal_spectrum_rmse': self.fitted_params.get('fractal_spectrum_rmse', 0.0),
            'fractal_fitting_success': self.fitted_params.get('fitting_success', False)
        }

# Convenience function for integration
def fit_and_visualize_fractal(image, image_path, output_dir="fractal_overlays"):
    """
    Complete fractal fitting and visualization pipeline
    
    Parameters:
    -----------
    image : np.ndarray
        Input grayscale image
    image_path : str
        Path to original image (for naming)
    output_dir : str
        Directory to save overlay visualizations
        
    Returns:
    --------
    tuple : (fractal_parameters_dict, overlay_image_path)
    """
    # Create fractal fitter
    fitter = FractalSurfaceFitter()
    
    # Fit fractal surface
    fractal_params = fitter.fit_fractal_surface(image)
    
    # Generate overlay visualization path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overlay_filename = f"{base_name}_fractal_overlay_{timestamp}.png"
    overlay_path = os.path.join(output_dir, overlay_filename)
    
    # Create visualization
    success = fitter.create_fractal_overlay(image, overlay_path)
    
    # Get CSV-ready parameters
    csv_params = fitter.get_fractal_parameters_for_csv()
    csv_params['fractal_overlay_path'] = overlay_path if success else ""
    
    return csv_params, overlay_path if success else None 