#!/usr/bin/env python3
"""
Ocean Thermal Analyzer
Segments ocean from land and enhances water temperature visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
from pathlib import Path
from scipy import ndimage
from skimage import filters, morphology, measure
import warnings
warnings.filterwarnings('ignore')


class OceanThermalSegmenter:
    """Segment ocean from land in thermal images"""
    
    def __init__(self, land_temp_threshold=19.0):
        """
        Initialize segmenter
        land_temp_threshold: Temperature in Celsius above which pixels are likely land
        """
        self.land_temp_threshold = land_temp_threshold
        
    def load_thermal_frame(self, frame_number, base_path="data/100MEDIA"):
        """Load and convert thermal data to temperature"""
        base_path = Path(base_path)
        
        # Load IRG or TIFF file
        irg_path = base_path / f"IRX_{frame_number:04d}.irg"
        tiff_path = base_path / f"IRX_{frame_number:04d}.TIFF"
        jpg_path = base_path / f"IRX_{frame_number:04d}.jpg"
        rgb_path = base_path / f"MAX_{frame_number:04d}.JPG"
        
        # Load thermal data
        if irg_path.exists():
            with open(irg_path, 'rb') as f:
                irg_data = f.read()
            
            # Parse raw thermal data (640x512 uint16 in deciKelvin)
            expected_pixels = 640 * 512
            pixel_data_size = expected_pixels * 2
            header_size = len(irg_data) - pixel_data_size
            
            if header_size > 0:
                raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
            else:
                raw_thermal = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
            
            raw_thermal = raw_thermal.reshape((512, 640))
        elif tiff_path.exists():
            tiff_img = Image.open(tiff_path)
            raw_thermal = np.array(tiff_img, dtype=np.uint16)
        else:
            raise FileNotFoundError(f"No thermal data found for frame {frame_number}")
        
        # Convert deciKelvin to Celsius
        temp_celsius = (raw_thermal / 10.0) - 273.15
        
        # Load JPG for reference
        jpg_img = None
        if jpg_path.exists():
            jpg_img = np.array(Image.open(jpg_path))
            if len(jpg_img.shape) == 3:
                jpg_img = np.dot(jpg_img[...,:3], [0.2989, 0.5870, 0.1140])
        
        # Load RGB for reference
        rgb_img = None
        if rgb_path.exists():
            rgb_img = np.array(Image.open(rgb_path))
        
        return {
            'raw': raw_thermal,
            'temp_celsius': temp_celsius,
            'jpg': jpg_img,
            'rgb': rgb_img,
            'frame_number': frame_number
        }
    
    def segment_ocean_land(self, temp_celsius, method='temperature'):
        """
        Segment ocean from land using various methods
        
        Methods:
        - 'temperature': Simple threshold based on temperature
        - 'variance': Use local variance (ocean has less variance)
        - 'combined': Combine temperature and variance
        - 'adaptive': Adaptive thresholding with morphology
        """
        
        if method == 'temperature':
            # Simple temperature threshold
            land_mask = temp_celsius > self.land_temp_threshold
            
        elif method == 'variance':
            # Calculate local variance (ocean has lower variance)
            local_variance = ndimage.generic_filter(temp_celsius, np.var, size=5)
            
            # Normalize variance
            var_norm = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min())
            
            # Ocean has low variance
            ocean_mask = var_norm < 0.3
            land_mask = ~ocean_mask
            
        elif method == 'combined':
            # Combine temperature and variance
            land_mask_temp = temp_celsius > self.land_temp_threshold
            
            # Local variance
            local_variance = ndimage.generic_filter(temp_celsius, np.var, size=5)
            var_norm = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min())
            
            # High variance areas
            high_var_mask = var_norm > 0.3
            
            # Combine: land is either hot OR has high variance
            land_mask = land_mask_temp | high_var_mask
            
        elif method == 'adaptive':
            # More sophisticated approach
            
            # 1. Initial temperature-based segmentation
            initial_land = temp_celsius > self.land_temp_threshold
            
            # 2. Calculate gradient magnitude (edges)
            grad_x = ndimage.sobel(temp_celsius, axis=1)
            grad_y = ndimage.sobel(temp_celsius, axis=0)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # 3. Calculate local statistics
            local_mean = ndimage.uniform_filter(temp_celsius, size=15)
            local_variance = ndimage.generic_filter(temp_celsius, np.var, size=15)
            
            # 4. Ocean characteristics: cooler, less variance, fewer edges
            ocean_score = np.zeros_like(temp_celsius)
            
            # Temperature score (ocean is cooler)
            temp_score = 1.0 - ((temp_celsius - temp_celsius.min()) / 
                               (temp_celsius.max() - temp_celsius.min()))
            
            # Variance score (ocean has less variance)
            var_score = 1.0 - ((local_variance - local_variance.min()) / 
                              (local_variance.max() - local_variance.min() + 1e-10))
            
            # Edge score (ocean has fewer edges)
            edge_score = 1.0 - ((gradient_mag - gradient_mag.min()) / 
                               (gradient_mag.max() - gradient_mag.min() + 1e-10))
            
            # Combine scores
            ocean_score = (temp_score * 0.4 + var_score * 0.3 + edge_score * 0.3)
            
            # Threshold to get ocean mask
            ocean_mask = ocean_score > 0.5
            
            # Clean up with morphology
            ocean_mask = morphology.binary_closing(ocean_mask, morphology.disk(3))
            ocean_mask = morphology.remove_small_objects(ocean_mask, min_size=1000)
            ocean_mask = morphology.remove_small_holes(ocean_mask, area_threshold=1000)
            
            land_mask = ~ocean_mask
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Clean up the mask
        land_mask = morphology.binary_closing(land_mask, morphology.disk(2))
        land_mask = morphology.remove_small_objects(land_mask, min_size=100)
        
        ocean_mask = ~land_mask
        
        return ocean_mask, land_mask
    
    def enhance_ocean_thermal(self, temp_celsius, ocean_mask):
        """
        Enhance thermal visualization for ocean areas
        """
        # Extract ocean temperatures
        ocean_temps = temp_celsius.copy()
        ocean_temps[~ocean_mask] = np.nan
        
        # Get valid ocean temperature range
        valid_temps = temp_celsius[ocean_mask]
        
        if len(valid_temps) == 0:
            return ocean_temps, 0, 0
        
        ocean_min = np.nanmin(valid_temps)
        ocean_max = np.nanmax(valid_temps)
        ocean_mean = np.nanmean(valid_temps)
        ocean_std = np.nanstd(valid_temps)
        
        print(f"Ocean temperature stats:")
        print(f"  Range: {ocean_min:.2f}°C to {ocean_max:.2f}°C")
        print(f"  Mean: {ocean_mean:.2f}°C, Std: {ocean_std:.2f}°C")
        
        # Create enhanced visualization
        # Rescale ocean temperatures to use full color range
        ocean_enhanced = ocean_temps.copy()
        ocean_enhanced = (ocean_enhanced - ocean_min) / (ocean_max - ocean_min + 1e-10)
        
        # Apply histogram equalization for better contrast
        ocean_flat = ocean_enhanced[ocean_mask]
        
        # Simple histogram equalization
        if len(ocean_flat) > 0:
            sorted_vals = np.sort(ocean_flat)
            cdf = np.arange(len(sorted_vals)) / float(len(sorted_vals))
            
            # Map values
            for i in range(len(ocean_flat)):
                idx = np.searchsorted(sorted_vals, ocean_flat[i])
                ocean_flat[i] = cdf[min(idx, len(cdf)-1)]
            
            ocean_enhanced[ocean_mask] = ocean_flat
        
        return ocean_enhanced, ocean_min, ocean_max
    
    def detect_ocean_features(self, ocean_temps, ocean_mask):
        """
        Detect interesting features in ocean thermal data
        """
        features = {}
        
        # Get valid ocean temperatures
        valid_temps = ocean_temps[ocean_mask]
        
        if len(valid_temps) == 0:
            return features
        
        ocean_mean = np.nanmean(valid_temps)
        ocean_std = np.nanstd(valid_temps)
        
        # Detect warm anomalies (potential currents or upwelling)
        warm_threshold = ocean_mean + 1.5 * ocean_std
        warm_anomalies = (ocean_temps > warm_threshold) & ocean_mask
        
        # Detect cold anomalies
        cold_threshold = ocean_mean - 1.5 * ocean_std
        cold_anomalies = (ocean_temps < cold_threshold) & ocean_mask
        
        # Find thermal fronts (high gradient areas)
        grad_x = ndimage.sobel(ocean_temps, axis=1)
        grad_y = ndimage.sobel(ocean_temps, axis=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mag[~ocean_mask] = 0
        
        # Threshold for fronts
        front_threshold = np.nanpercentile(gradient_mag[ocean_mask], 90)
        thermal_fronts = (gradient_mag > front_threshold) & ocean_mask
        
        features['warm_anomalies'] = warm_anomalies
        features['cold_anomalies'] = cold_anomalies
        features['thermal_fronts'] = thermal_fronts
        features['gradient_magnitude'] = gradient_mag
        
        return features


class OceanThermalViewer:
    """Interactive viewer for ocean thermal analysis"""
    
    def __init__(self, base_path="data/100MEDIA"):
        self.segmenter = OceanThermalSegmenter()
        self.base_path = Path(base_path)
        self.current_frame = None
        
    def process_single_frame(self, frame_number, method='adaptive'):
        """Process a single frame"""
        
        # Load thermal data
        data = self.segmenter.load_thermal_frame(frame_number, self.base_path)
        temp_celsius = data['temp_celsius']
        
        # Segment ocean from land
        ocean_mask, land_mask = self.segmenter.segment_ocean_land(temp_celsius, method=method)
        
        # Enhance ocean thermal
        ocean_enhanced, ocean_min, ocean_max = self.segmenter.enhance_ocean_thermal(
            temp_celsius, ocean_mask)
        
        # Detect features
        features = self.segmenter.detect_ocean_features(temp_celsius, ocean_mask)
        
        return {
            'frame_number': frame_number,
            'temp_celsius': temp_celsius,
            'ocean_mask': ocean_mask,
            'land_mask': land_mask,
            'ocean_enhanced': ocean_enhanced,
            'ocean_min': ocean_min,
            'ocean_max': ocean_max,
            'features': features,
            'jpg': data['jpg'],
            'rgb': data['rgb']
        }
    
    def visualize_frame(self, result):
        """Visualize segmentation and enhancement results"""
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(f'Ocean Thermal Analysis - Frame {result["frame_number"]:04d}', fontsize=14)
        
        # Row 1: Original and segmentation
        if result['rgb'] is not None:
            axes[0, 0].imshow(result['rgb'])
            axes[0, 0].set_title('RGB Image')
        else:
            axes[0, 0].text(0.5, 0.5, 'No RGB', ha='center', va='center')
        axes[0, 0].axis('off')
        
        # Original thermal
        im1 = axes[0, 1].imshow(result['temp_celsius'], cmap='RdYlBu_r')
        axes[0, 1].set_title(f'Original Thermal (°C)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # Segmentation masks
        mask_display = np.zeros((*result['ocean_mask'].shape, 3))
        mask_display[result['ocean_mask']] = [0, 0, 1]  # Blue for ocean
        mask_display[result['land_mask']] = [0.5, 0.3, 0]  # Brown for land
        
        axes[0, 2].imshow(mask_display)
        axes[0, 2].set_title('Segmentation\n(Blue=Ocean, Brown=Land)')
        axes[0, 2].axis('off')
        
        # Masked thermal (ocean only)
        ocean_only = result['temp_celsius'].copy()
        ocean_only[~result['ocean_mask']] = np.nan
        
        im2 = axes[0, 3].imshow(ocean_only, cmap='viridis')
        axes[0, 3].set_title(f'Ocean Only\n[{result["ocean_min"]:.2f}, {result["ocean_max"]:.2f}]°C')
        axes[0, 3].axis('off')
        plt.colorbar(im2, ax=axes[0, 3], fraction=0.046)
        
        # Row 2: Enhanced and features
        # Enhanced ocean visualization
        im3 = axes[1, 0].imshow(result['ocean_enhanced'], cmap='plasma')
        axes[1, 0].set_title('Enhanced Ocean\n(Rescaled & Equalized)')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Temperature histogram
        ocean_temps = result['temp_celsius'][result['ocean_mask']]
        land_temps = result['temp_celsius'][result['land_mask']]
        
        axes[1, 1].hist(ocean_temps, bins=50, alpha=0.5, label='Ocean', color='blue', density=True)
        axes[1, 1].hist(land_temps, bins=50, alpha=0.5, label='Land', color='brown', density=True)
        axes[1, 1].set_xlabel('Temperature (°C)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Temperature Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Features
        if 'features' in result and result['features']:
            feature_display = np.zeros((*result['ocean_mask'].shape, 3))
            feature_display[result['ocean_mask']] = [0, 0, 0.3]  # Dark blue base
            
            if 'warm_anomalies' in result['features']:
                feature_display[result['features']['warm_anomalies']] = [1, 0, 0]  # Red for warm
            
            if 'cold_anomalies' in result['features']:
                feature_display[result['features']['cold_anomalies']] = [0, 0, 1]  # Blue for cold
            
            if 'thermal_fronts' in result['features']:
                feature_display[result['features']['thermal_fronts']] = [1, 1, 0]  # Yellow for fronts
            
            axes[1, 2].imshow(feature_display)
            axes[1, 2].set_title('Ocean Features\n(Red=Warm, Blue=Cold, Yellow=Fronts)')
            axes[1, 2].axis('off')
        
        # Gradient magnitude for ocean
        if 'features' in result and 'gradient_magnitude' in result['features']:
            grad_ocean = result['features']['gradient_magnitude'].copy()
            grad_ocean[~result['ocean_mask']] = np.nan
            
            im4 = axes[1, 3].imshow(grad_ocean, cmap='hot')
            axes[1, 3].set_title('Thermal Gradients\n(Ocean Only)')
            axes[1, 3].axis('off')
            plt.colorbar(im4, ax=axes[1, 3], fraction=0.046)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_viewer(self):
        """Create interactive viewer with controls"""
        
        # Get available frames
        irg_files = list(self.base_path.glob("IRX_*.irg"))
        frame_numbers = sorted([int(f.stem.split('_')[1]) for f in irg_files])
        
        if not frame_numbers:
            print("No thermal files found!")
            return
        
        print(f"Found {len(frame_numbers)} frames")
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Current settings
        current = {
            'frame_idx': 0,
            'method': 'adaptive',
            'temp_threshold': 19.0
        }
        
        # Process initial frame
        self.segmenter.land_temp_threshold = current['temp_threshold']
        result = self.process_single_frame(frame_numbers[0], current['method'])
        
        # Create layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Image displays
        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_thermal = fig.add_subplot(gs[0, 1])
        ax_mask = fig.add_subplot(gs[0, 2])
        ax_ocean = fig.add_subplot(gs[0, 3])
        ax_enhanced = fig.add_subplot(gs[1, 0])
        ax_hist = fig.add_subplot(gs[1, 1])
        ax_features = fig.add_subplot(gs[1, 2])
        ax_gradient = fig.add_subplot(gs[1, 3])
        ax_profile = fig.add_subplot(gs[2, :2])
        ax_stats = fig.add_subplot(gs[2, 2:])
        
        def update_display(val=None):
            """Update all displays"""
            
            # Clear axes
            for ax in [ax_rgb, ax_thermal, ax_mask, ax_ocean, ax_enhanced,
                      ax_hist, ax_features, ax_gradient, ax_profile, ax_stats]:
                ax.clear()
            
            # Get current frame
            frame_num = frame_numbers[current['frame_idx']]
            
            # Process frame
            self.segmenter.land_temp_threshold = current['temp_threshold']
            result = self.process_single_frame(frame_num, current['method'])
            
            fig.suptitle(f'Ocean Thermal Analysis - Frame {frame_num:04d}', fontsize=14)
            
            # Display images
            if result['rgb'] is not None:
                ax_rgb.imshow(result['rgb'])
            ax_rgb.set_title('RGB')
            ax_rgb.axis('off')
            
            im = ax_thermal.imshow(result['temp_celsius'], cmap='RdYlBu_r')
            ax_thermal.set_title('Thermal (°C)')
            ax_thermal.axis('off')
            
            # Mask
            mask_display = np.zeros((*result['ocean_mask'].shape, 3))
            mask_display[result['ocean_mask']] = [0, 0, 1]
            mask_display[result['land_mask']] = [0.5, 0.3, 0]
            ax_mask.imshow(mask_display)
            ax_mask.set_title('Segmentation')
            ax_mask.axis('off')
            
            # Ocean only
            ocean_only = result['temp_celsius'].copy()
            ocean_only[~result['ocean_mask']] = np.nan
            ax_ocean.imshow(ocean_only, cmap='viridis')
            ax_ocean.set_title(f'Ocean [{result["ocean_min"]:.1f}-{result["ocean_max"]:.1f}°C]')
            ax_ocean.axis('off')
            
            # Enhanced
            ax_enhanced.imshow(result['ocean_enhanced'], cmap='plasma')
            ax_enhanced.set_title('Enhanced Ocean')
            ax_enhanced.axis('off')
            
            # Histogram
            ocean_temps = result['temp_celsius'][result['ocean_mask']]
            land_temps = result['temp_celsius'][result['land_mask']]
            
            if len(ocean_temps) > 0:
                ax_hist.hist(ocean_temps, bins=30, alpha=0.7, label='Ocean', color='blue')
            if len(land_temps) > 0:
                ax_hist.hist(land_temps, bins=30, alpha=0.7, label='Land', color='brown')
            ax_hist.set_xlabel('Temperature (°C)')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            
            # Features
            if result['features']:
                feature_display = np.zeros((*result['ocean_mask'].shape, 3))
                feature_display[result['ocean_mask']] = [0, 0, 0.3]
                
                if 'warm_anomalies' in result['features']:
                    feature_display[result['features']['warm_anomalies']] = [1, 0, 0]
                if 'cold_anomalies' in result['features']:
                    feature_display[result['features']['cold_anomalies']] = [0, 0, 1]
                
                ax_features.imshow(feature_display)
            ax_features.set_title('Anomalies')
            ax_features.axis('off')
            
            # Gradient
            if result['features'] and 'gradient_magnitude' in result['features']:
                grad_ocean = result['features']['gradient_magnitude'].copy()
                grad_ocean[~result['ocean_mask']] = np.nan
                ax_gradient.imshow(grad_ocean, cmap='hot')
            ax_gradient.set_title('Gradients')
            ax_gradient.axis('off')
            
            # Profile
            center_y = result['temp_celsius'].shape[0] // 2
            temps_line = result['temp_celsius'][center_y, :]
            ocean_line = result['ocean_mask'][center_y, :]
            
            x = np.arange(len(temps_line))
            ax_profile.plot(x[ocean_line], temps_line[ocean_line], 'b-', label='Ocean', alpha=0.7)
            ax_profile.plot(x[~ocean_line], temps_line[~ocean_line], 'r-', label='Land', alpha=0.7)
            ax_profile.set_xlabel('Position')
            ax_profile.set_ylabel('Temperature (°C)')
            ax_profile.set_title('Horizontal Profile (Center)')
            ax_profile.legend()
            ax_profile.grid(True, alpha=0.3)
            
            # Stats
            stats_text = f"Frame {frame_num:04d} Statistics:\n\n"
            stats_text += f"Ocean Coverage: {100*result['ocean_mask'].sum()/result['ocean_mask'].size:.1f}%\n"
            
            if len(ocean_temps) > 0:
                stats_text += f"\nOcean Temperature:\n"
                stats_text += f"  Min: {ocean_temps.min():.2f}°C\n"
                stats_text += f"  Max: {ocean_temps.max():.2f}°C\n"
                stats_text += f"  Mean: {ocean_temps.mean():.2f}°C\n"
                stats_text += f"  Std: {ocean_temps.std():.2f}°C\n"
            
            if len(land_temps) > 0:
                stats_text += f"\nLand Temperature:\n"
                stats_text += f"  Min: {land_temps.min():.2f}°C\n"
                stats_text += f"  Max: {land_temps.max():.2f}°C\n"
                stats_text += f"  Mean: {land_temps.mean():.2f}°C\n"
            
            stats_text += f"\nMethod: {current['method']}\n"
            stats_text += f"Threshold: {current['temp_threshold']:.1f}°C"
            
            ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                         fontsize=10, verticalalignment='center')
            ax_stats.axis('off')
            
            fig.canvas.draw_idle()
        
        # Add controls
        ax_slider = plt.axes([0.1, 0.02, 0.4, 0.02])
        slider = Slider(ax_slider, 'Frame', 0, len(frame_numbers)-1,
                       valinit=0, valstep=1)
        
        ax_threshold = plt.axes([0.6, 0.02, 0.3, 0.02])
        threshold_slider = Slider(ax_threshold, 'Land Temp (°C)', 15, 25,
                                valinit=19, valstep=0.5)
        
        def update_frame(val):
            current['frame_idx'] = int(slider.val)
            update_display()
        
        def update_threshold(val):
            current['temp_threshold'] = threshold_slider.val
            update_display()
        
        slider.on_changed(update_frame)
        threshold_slider.on_changed(update_threshold)
        
        # Method buttons
        ax_btn1 = plt.axes([0.05, 0.06, 0.1, 0.03])
        btn1 = Button(ax_btn1, 'Temperature')
        
        ax_btn2 = plt.axes([0.16, 0.06, 0.1, 0.03])
        btn2 = Button(ax_btn2, 'Variance')
        
        ax_btn3 = plt.axes([0.27, 0.06, 0.1, 0.03])
        btn3 = Button(ax_btn3, 'Combined')
        
        ax_btn4 = plt.axes([0.38, 0.06, 0.1, 0.03])
        btn4 = Button(ax_btn4, 'Adaptive')
        
        def set_method_temp(event):
            current['method'] = 'temperature'
            update_display()
        
        def set_method_var(event):
            current['method'] = 'variance'
            update_display()
        
        def set_method_combined(event):
            current['method'] = 'combined'
            update_display()
        
        def set_method_adaptive(event):
            current['method'] = 'adaptive'
            update_display()
        
        btn1.on_clicked(set_method_temp)
        btn2.on_clicked(set_method_var)
        btn3.on_clicked(set_method_combined)
        btn4.on_clicked(set_method_adaptive)
        
        # Initial display
        update_display()
        
        plt.show()
        
        return fig


def main():
    """Main entry point"""
    print("Ocean Thermal Analyzer")
    print("=" * 40)
    
    viewer = OceanThermalViewer()
    
    print("\nLaunching interactive viewer...")
    print("Controls:")
    print("  - Frame slider: Navigate between frames")
    print("  - Land Temp slider: Adjust temperature threshold for land")
    print("  - Method buttons: Choose segmentation algorithm")
    print("\nClose window to exit")
    
    viewer.create_interactive_viewer()


if __name__ == "__main__":
    main()