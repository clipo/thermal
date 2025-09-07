#!/usr/bin/env python3
"""
RGB-Based Ocean Segmentation
Uses RGB drone images to create accurate land/ocean/wave masks for thermal analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RangeSlider
from PIL import Image
from pathlib import Path
from scipy import ndimage
from skimage import filters, morphology, measure, color
import warnings
warnings.filterwarnings('ignore')


class RGBOceanSegmenter:
    """Use RGB images to segment ocean, land, and waves"""
    
    def __init__(self):
        """Initialize segmenter with color thresholds"""
        # Default color ranges for detection (can be tuned)
        self.ocean_hsv_range = {
            'h_min': 180, 'h_max': 250,  # Blue hues (in degrees)
            's_min': 20, 's_max': 255,   # Some saturation
            'v_min': 20, 'v_max': 200    # Not too bright (excludes white foam)
        }
        
        self.land_hsv_range = {
            'h_min': 40, 'h_max': 150,   # Green to yellow-green hues
            's_min': 15, 's_max': 255,   # Any saturation
            'v_min': 10, 'v_max': 255    # Any value
        }
        
        self.wave_hsv_range = {
            'h_min': 0, 'h_max': 360,     # Any hue
            's_min': 0, 's_max': 30,      # Very low saturation (white/gray)
            'v_min': 180, 'v_max': 255   # High brightness (white foam)
        }
        
    def load_rgb_thermal_pair(self, frame_number, base_path="data/100MEDIA"):
        """Load RGB and thermal data for a frame"""
        base_path = Path(base_path)
        
        # Load RGB
        rgb_path = base_path / f"MAX_{frame_number:04d}.JPG"
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB file not found: {rgb_path}")
        
        rgb_img = np.array(Image.open(rgb_path))
        
        # Load thermal
        irg_path = base_path / f"IRX_{frame_number:04d}.irg"
        tiff_path = base_path / f"IRX_{frame_number:04d}.TIFF"
        
        if irg_path.exists():
            with open(irg_path, 'rb') as f:
                irg_data = f.read()
            
            expected_pixels = 640 * 512
            pixel_data_size = expected_pixels * 2
            header_size = len(irg_data) - pixel_data_size
            
            if header_size > 0:
                raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
            else:
                raw_thermal = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
            
            raw_thermal = raw_thermal.reshape((512, 640))
        elif tiff_path.exists():
            raw_thermal = np.array(Image.open(tiff_path), dtype=np.uint16)
        else:
            raise FileNotFoundError(f"No thermal data for frame {frame_number}")
        
        # Convert to Celsius
        temp_celsius = (raw_thermal / 10.0) - 273.15
        
        # Resize RGB to match thermal dimensions if needed
        if rgb_img.shape[:2] != temp_celsius.shape:
            rgb_resized = np.array(Image.fromarray(rgb_img).resize(
                (temp_celsius.shape[1], temp_celsius.shape[0]), 
                Image.Resampling.BILINEAR
            ))
        else:
            rgb_resized = rgb_img
        
        return {
            'rgb_original': rgb_img,
            'rgb_resized': rgb_resized,
            'thermal': temp_celsius,
            'raw_thermal': raw_thermal,
            'frame_number': frame_number
        }
    
    def segment_rgb_image(self, rgb_img, method='hsv'):
        """
        Segment RGB image into ocean, land, and wave zones
        
        Methods:
        - 'hsv': HSV color space thresholding
        - 'lab': LAB color space analysis
        - 'combined': Multiple color spaces
        """
        
        if method == 'hsv':
            return self._segment_hsv(rgb_img)
        elif method == 'lab':
            return self._segment_lab(rgb_img)
        elif method == 'combined':
            return self._segment_combined(rgb_img)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _segment_hsv(self, rgb_img):
        """HSV-based segmentation"""
        
        # Convert to HSV
        hsv = color.rgb2hsv(rgb_img)
        h = hsv[:, :, 0] * 360  # Convert to degrees
        s = hsv[:, :, 1] * 255
        v = hsv[:, :, 2] * 255
        
        # Detect ocean (blue colors)
        ocean_mask = (
            (h >= self.ocean_hsv_range['h_min']) & 
            (h <= self.ocean_hsv_range['h_max']) &
            (s >= self.ocean_hsv_range['s_min']) & 
            (s <= self.ocean_hsv_range['s_max']) &
            (v >= self.ocean_hsv_range['v_min']) & 
            (v <= self.ocean_hsv_range['v_max'])
        )
        
        # Detect land (green/brown colors)
        land_mask = (
            (h >= self.land_hsv_range['h_min']) & 
            (h <= self.land_hsv_range['h_max']) &
            (s >= self.land_hsv_range['s_min']) & 
            (s <= self.land_hsv_range['s_max']) &
            (v >= self.land_hsv_range['v_min']) & 
            (v <= self.land_hsv_range['v_max'])
        )
        
        # Detect waves/foam (white/bright areas)
        wave_mask = (
            (s <= self.wave_hsv_range['s_max']) &
            (v >= self.wave_hsv_range['v_min'])
        )
        
        # Also detect very dark areas as potential land/shadows
        dark_mask = (v < 30)
        land_mask = land_mask | dark_mask
        
        # Clean up masks
        ocean_mask = self._clean_mask(ocean_mask)
        land_mask = self._clean_mask(land_mask)
        wave_mask = self._clean_mask(wave_mask, min_size=50)
        
        # Resolve conflicts (wave takes priority, then land, then ocean)
        land_mask = land_mask & ~wave_mask
        ocean_mask = ocean_mask & ~wave_mask & ~land_mask
        
        return {
            'ocean': ocean_mask,
            'land': land_mask,
            'waves': wave_mask,
            'hsv': hsv
        }
    
    def _segment_lab(self, rgb_img):
        """LAB color space segmentation"""
        
        # Convert to LAB
        lab = color.rgb2lab(rgb_img)
        l = lab[:, :, 0]  # Lightness
        a = lab[:, :, 1]  # Green-Red
        b = lab[:, :, 2]  # Blue-Yellow
        
        # Ocean: blue (negative b values)
        ocean_mask = (b < -5) & (l > 20) & (l < 80)
        
        # Land: green (negative a values) or brown/dark
        land_mask = ((a < -5) & (b > -10)) | (l < 30)
        
        # Waves: very bright
        wave_mask = (l > 85) & (np.abs(a) < 10) & (np.abs(b) < 10)
        
        # Clean up
        ocean_mask = self._clean_mask(ocean_mask)
        land_mask = self._clean_mask(land_mask)
        wave_mask = self._clean_mask(wave_mask, min_size=50)
        
        # Resolve conflicts
        land_mask = land_mask & ~wave_mask
        ocean_mask = ocean_mask & ~wave_mask & ~land_mask
        
        return {
            'ocean': ocean_mask,
            'land': land_mask,
            'waves': wave_mask,
            'lab': lab
        }
    
    def _segment_combined(self, rgb_img):
        """Combined multi-color-space segmentation"""
        
        # Get both HSV and LAB segmentations
        hsv_result = self._segment_hsv(rgb_img)
        lab_result = self._segment_lab(rgb_img)
        
        # Combine with voting
        ocean_votes = hsv_result['ocean'].astype(int) + lab_result['ocean'].astype(int)
        land_votes = hsv_result['land'].astype(int) + lab_result['land'].astype(int)
        wave_votes = hsv_result['waves'].astype(int) + lab_result['waves'].astype(int)
        
        # Threshold (at least one method must agree)
        ocean_mask = ocean_votes >= 1
        land_mask = land_votes >= 1
        wave_mask = wave_votes >= 2  # Waves need stronger agreement
        
        # Additional color analysis
        # Check for specific colors
        r, g, b = rgb_img[:,:,0], rgb_img[:,:,1], rgb_img[:,:,2]
        
        # Strong blue indication (B > R and B > G)
        strong_blue = (b > r * 1.2) & (b > g * 1.1) & (b > 50)
        ocean_mask = ocean_mask | strong_blue
        
        # Strong green indication
        strong_green = (g > r * 1.1) & (g > b * 1.1) & (g > 30)
        land_mask = land_mask | strong_green
        
        # Clean up final masks
        ocean_mask = self._clean_mask(ocean_mask, min_size=500)
        land_mask = self._clean_mask(land_mask, min_size=500)
        wave_mask = self._clean_mask(wave_mask, min_size=100)
        
        # Resolve conflicts with priority
        wave_mask = wave_mask  # Waves highest priority
        land_mask = land_mask & ~wave_mask
        ocean_mask = ocean_mask & ~wave_mask & ~land_mask
        
        # Fill undefined areas based on neighbors
        undefined = ~(ocean_mask | land_mask | wave_mask)
        if undefined.sum() > 0:
            # Use dilation to assign undefined pixels
            ocean_dilated = morphology.binary_dilation(ocean_mask, morphology.disk(5))
            land_dilated = morphology.binary_dilation(land_mask, morphology.disk(5))
            
            # Assign undefined pixels to nearest category
            undefined_to_ocean = undefined & ocean_dilated & ~land_dilated
            undefined_to_land = undefined & land_dilated & ~ocean_dilated
            
            ocean_mask = ocean_mask | undefined_to_ocean
            land_mask = land_mask | undefined_to_land
        
        return {
            'ocean': ocean_mask,
            'land': land_mask,
            'waves': wave_mask
        }
    
    def _clean_mask(self, mask, min_size=200):
        """Clean up a binary mask"""
        # Remove small objects
        mask = morphology.remove_small_objects(mask, min_size=min_size)
        
        # Fill small holes
        mask = morphology.remove_small_holes(mask, area_threshold=min_size)
        
        # Smooth boundaries
        mask = morphology.binary_closing(mask, morphology.disk(2))
        mask = morphology.binary_opening(mask, morphology.disk(1))
        
        return mask
    
    def apply_mask_to_thermal(self, thermal, masks, exclude_waves=True):
        """Apply RGB-derived masks to thermal data"""
        
        ocean_thermal = thermal.copy()
        land_thermal = thermal.copy()
        
        # Create excluded mask
        if exclude_waves and 'waves' in masks:
            excluded = masks['waves']
            ocean_thermal[~masks['ocean'] | excluded] = np.nan
        else:
            ocean_thermal[~masks['ocean']] = np.nan
        
        land_thermal[~masks['land']] = np.nan
        
        # Calculate statistics
        ocean_valid = thermal[masks['ocean']]
        land_valid = thermal[masks['land']]
        
        if 'waves' in masks:
            wave_valid = thermal[masks['waves']]
        else:
            wave_valid = np.array([])
        
        stats = {
            'ocean_mean': np.mean(ocean_valid) if len(ocean_valid) > 0 else np.nan,
            'ocean_std': np.std(ocean_valid) if len(ocean_valid) > 0 else np.nan,
            'ocean_min': np.min(ocean_valid) if len(ocean_valid) > 0 else np.nan,
            'ocean_max': np.max(ocean_valid) if len(ocean_valid) > 0 else np.nan,
            'land_mean': np.mean(land_valid) if len(land_valid) > 0 else np.nan,
            'land_std': np.std(land_valid) if len(land_valid) > 0 else np.nan,
            'land_min': np.min(land_valid) if len(land_valid) > 0 else np.nan,
            'land_max': np.max(land_valid) if len(land_valid) > 0 else np.nan,
            'wave_mean': np.mean(wave_valid) if len(wave_valid) > 0 else np.nan,
            'wave_std': np.std(wave_valid) if len(wave_valid) > 0 else np.nan,
        }
        
        return ocean_thermal, land_thermal, stats
    
    def enhance_ocean_visualization(self, ocean_thermal, masks):
        """Enhance ocean thermal visualization"""
        
        valid_ocean = ocean_thermal[masks['ocean']]
        
        if len(valid_ocean) == 0:
            return ocean_thermal
        
        # Get range
        ocean_min = np.nanmin(valid_ocean)
        ocean_max = np.nanmax(valid_ocean)
        
        # Normalize to 0-1
        ocean_enhanced = (ocean_thermal - ocean_min) / (ocean_max - ocean_min + 1e-10)
        
        # Apply histogram equalization
        ocean_flat = ocean_enhanced[masks['ocean']]
        sorted_vals = np.sort(ocean_flat)
        cdf = np.arange(len(sorted_vals)) / float(len(sorted_vals))
        
        # Map values
        for i in range(len(ocean_flat)):
            idx = np.searchsorted(sorted_vals, ocean_flat[i])
            ocean_flat[i] = cdf[min(idx, len(cdf)-1)]
        
        ocean_enhanced[masks['ocean']] = ocean_flat
        ocean_enhanced[~masks['ocean']] = np.nan
        
        return ocean_enhanced


class RGBOceanViewer:
    """Interactive viewer for RGB-based ocean segmentation"""
    
    def __init__(self, base_path="data/100MEDIA"):
        self.segmenter = RGBOceanSegmenter()
        self.base_path = Path(base_path)
        
    def create_interactive_viewer(self):
        """Create interactive viewer with RGB-based segmentation"""
        
        # Get available frames with both RGB and thermal
        rgb_files = list(self.base_path.glob("MAX_*.JPG"))
        frame_numbers = []
        
        for rgb_file in sorted(rgb_files):
            frame_num = int(rgb_file.stem.split('_')[1])
            # Check if thermal exists
            if (self.base_path / f"IRX_{frame_num:04d}.irg").exists() or \
               (self.base_path / f"IRX_{frame_num:04d}.TIFF").exists():
                frame_numbers.append(frame_num)
        
        if not frame_numbers:
            print("No matching RGB-thermal pairs found!")
            return
        
        print(f"Found {len(frame_numbers)} RGB-thermal pairs")
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # State
        current = {
            'frame_idx': 0,
            'method': 'combined',
            'exclude_waves': True
        }
        
        # Process initial frame
        data = self.segmenter.load_rgb_thermal_pair(frame_numbers[0], self.base_path)
        masks = self.segmenter.segment_rgb_image(data['rgb_resized'], current['method'])
        
        # Create layout
        gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)
        
        # Axes
        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_hsv = fig.add_subplot(gs[0, 1])
        ax_masks = fig.add_subplot(gs[0, 2])
        ax_thermal = fig.add_subplot(gs[0, 3])
        ax_ocean_mask = fig.add_subplot(gs[0, 4])
        
        ax_ocean_thermal = fig.add_subplot(gs[1, 0])
        ax_land_thermal = fig.add_subplot(gs[1, 1])
        ax_wave_thermal = fig.add_subplot(gs[1, 2])
        ax_enhanced = fig.add_subplot(gs[1, 3])
        ax_hist = fig.add_subplot(gs[1, 4])
        
        ax_profile = fig.add_subplot(gs[2, :3])
        ax_stats = fig.add_subplot(gs[2, 3:])
        
        def update_display(val=None):
            """Update display"""
            
            # Clear all axes
            for ax in [ax_rgb, ax_hsv, ax_masks, ax_thermal, ax_ocean_mask,
                      ax_ocean_thermal, ax_land_thermal, ax_wave_thermal,
                      ax_enhanced, ax_hist, ax_profile, ax_stats]:
                ax.clear()
            
            # Get current frame
            frame_num = frame_numbers[current['frame_idx']]
            
            # Load data
            data = self.segmenter.load_rgb_thermal_pair(frame_num, self.base_path)
            
            # Segment RGB
            masks = self.segmenter.segment_rgb_image(data['rgb_resized'], current['method'])
            
            # Apply to thermal
            ocean_thermal, land_thermal, stats = self.segmenter.apply_mask_to_thermal(
                data['thermal'], masks, current['exclude_waves']
            )
            
            fig.suptitle(f'RGB-Based Ocean Segmentation - Frame {frame_num:04d}', fontsize=14)
            
            # Row 1: RGB analysis
            ax_rgb.imshow(data['rgb_resized'])
            ax_rgb.set_title('RGB Image')
            ax_rgb.axis('off')
            
            # Show HSV if available
            if 'hsv' in masks:
                ax_hsv.imshow(masks['hsv'])
                ax_hsv.set_title('HSV Color Space')
            elif 'lab' in masks:
                # Normalize LAB for display
                lab_display = masks['lab'].copy()
                lab_display[:,:,0] = lab_display[:,:,0] / 100  # L: 0-100
                lab_display[:,:,1] = (lab_display[:,:,1] + 128) / 255  # a: -128 to 127
                lab_display[:,:,2] = (lab_display[:,:,2] + 128) / 255  # b: -128 to 127
                ax_hsv.imshow(lab_display)
                ax_hsv.set_title('LAB Color Space')
            ax_hsv.axis('off')
            
            # Segmentation masks
            mask_display = np.zeros((*masks['ocean'].shape, 3))
            mask_display[masks['ocean']] = [0, 0.3, 1]  # Blue for ocean
            mask_display[masks['land']] = [0, 0.5, 0]   # Green for land
            if 'waves' in masks:
                mask_display[masks['waves']] = [1, 1, 1]  # White for waves
            
            ax_masks.imshow(mask_display)
            ax_masks.set_title('Segmentation\n(Blue=Ocean, Green=Land, White=Waves)')
            ax_masks.axis('off')
            
            # Original thermal
            im = ax_thermal.imshow(data['thermal'], cmap='RdYlBu_r')
            ax_thermal.set_title('Original Thermal')
            ax_thermal.axis('off')
            
            # Ocean mask overlay
            overlay = data['rgb_resized'].copy()
            overlay[~masks['ocean']] = overlay[~masks['ocean']] * 0.3  # Darken non-ocean
            ax_ocean_mask.imshow(overlay)
            ax_ocean_mask.set_title('Ocean Mask on RGB')
            ax_ocean_mask.axis('off')
            
            # Row 2: Thermal analysis
            im1 = ax_ocean_thermal.imshow(ocean_thermal, cmap='viridis')
            ax_ocean_thermal.set_title(f'Ocean Thermal\n[{stats["ocean_min"]:.1f}-{stats["ocean_max"]:.1f}°C]')
            ax_ocean_thermal.axis('off')
            
            im2 = ax_land_thermal.imshow(land_thermal, cmap='terrain')
            ax_land_thermal.set_title(f'Land Thermal\n[{stats["land_min"]:.1f}-{stats["land_max"]:.1f}°C]')
            ax_land_thermal.axis('off')
            
            # Wave zone thermal
            if 'waves' in masks:
                wave_thermal = data['thermal'].copy()
                wave_thermal[~masks['waves']] = np.nan
                im3 = ax_wave_thermal.imshow(wave_thermal, cmap='coolwarm')
                ax_wave_thermal.set_title(f'Wave Zone\nMean: {stats["wave_mean"]:.1f}°C')
            else:
                ax_wave_thermal.text(0.5, 0.5, 'No waves detected', 
                                    ha='center', va='center', transform=ax_wave_thermal.transAxes)
            ax_wave_thermal.axis('off')
            
            # Enhanced ocean
            ocean_enhanced = self.segmenter.enhance_ocean_visualization(ocean_thermal, masks)
            im4 = ax_enhanced.imshow(ocean_enhanced, cmap='plasma')
            ax_enhanced.set_title('Enhanced Ocean')
            ax_enhanced.axis('off')
            
            # Histogram
            ocean_temps = data['thermal'][masks['ocean']]
            land_temps = data['thermal'][masks['land']]
            
            if len(ocean_temps) > 0:
                ax_hist.hist(ocean_temps, bins=30, alpha=0.7, label=f'Ocean (μ={stats["ocean_mean"]:.1f})', 
                           color='blue', density=True)
            if len(land_temps) > 0:
                ax_hist.hist(land_temps, bins=30, alpha=0.7, label=f'Land (μ={stats["land_mean"]:.1f})', 
                           color='green', density=True)
            if 'waves' in masks:
                wave_temps = data['thermal'][masks['waves']]
                if len(wave_temps) > 0:
                    ax_hist.hist(wave_temps, bins=30, alpha=0.7, label=f'Waves (μ={stats["wave_mean"]:.1f})', 
                               color='cyan', density=True)
            
            ax_hist.set_xlabel('Temperature (°C)')
            ax_hist.set_ylabel('Density')
            ax_hist.legend(fontsize=8)
            ax_hist.grid(True, alpha=0.3)
            
            # Profile
            center_y = data['thermal'].shape[0] // 2
            temps_line = data['thermal'][center_y, :]
            ocean_line = masks['ocean'][center_y, :]
            land_line = masks['land'][center_y, :]
            
            x = np.arange(len(temps_line))
            ax_profile.plot(x[ocean_line], temps_line[ocean_line], 'b.', label='Ocean', alpha=0.5)
            ax_profile.plot(x[land_line], temps_line[land_line], 'g.', label='Land', alpha=0.5)
            
            if 'waves' in masks:
                wave_line = masks['waves'][center_y, :]
                ax_profile.plot(x[wave_line], temps_line[wave_line], 'c.', label='Waves', alpha=0.5)
            
            ax_profile.set_xlabel('Position')
            ax_profile.set_ylabel('Temperature (°C)')
            ax_profile.set_title('Horizontal Profile (Center Line)')
            ax_profile.legend()
            ax_profile.grid(True, alpha=0.3)
            
            # Statistics
            stats_text = f"Frame {frame_num:04d} Statistics:\n\n"
            stats_text += f"Coverage:\n"
            total_pixels = masks['ocean'].size
            stats_text += f"  Ocean: {100*masks['ocean'].sum()/total_pixels:.1f}%\n"
            stats_text += f"  Land: {100*masks['land'].sum()/total_pixels:.1f}%\n"
            if 'waves' in masks:
                stats_text += f"  Waves: {100*masks['waves'].sum()/total_pixels:.1f}%\n"
            
            stats_text += f"\nTemperatures:\n"
            if not np.isnan(stats['ocean_mean']):
                stats_text += f"  Ocean: {stats['ocean_mean']:.1f}±{stats['ocean_std']:.1f}°C\n"
            if not np.isnan(stats['land_mean']):
                stats_text += f"  Land: {stats['land_mean']:.1f}±{stats['land_std']:.1f}°C\n"
            if not np.isnan(stats.get('wave_mean', np.nan)):
                stats_text += f"  Waves: {stats['wave_mean']:.1f}±{stats['wave_std']:.1f}°C\n"
            
            stats_text += f"\nSettings:\n"
            stats_text += f"  Method: {current['method']}\n"
            stats_text += f"  Exclude waves: {current['exclude_waves']}"
            
            ax_stats.text(0.05, 0.5, stats_text, transform=ax_stats.transAxes,
                         fontsize=9, verticalalignment='center')
            ax_stats.axis('off')
            
            fig.canvas.draw_idle()
        
        # Controls
        ax_slider = plt.axes([0.1, 0.02, 0.5, 0.02])
        slider = Slider(ax_slider, 'Frame', 0, len(frame_numbers)-1,
                       valinit=0, valstep=1)
        
        def update_frame(val):
            current['frame_idx'] = int(slider.val)
            update_display()
        
        slider.on_changed(update_frame)
        
        # Method buttons
        ax_btn_hsv = plt.axes([0.7, 0.02, 0.06, 0.025])
        btn_hsv = Button(ax_btn_hsv, 'HSV')
        
        ax_btn_lab = plt.axes([0.77, 0.02, 0.06, 0.025])
        btn_lab = Button(ax_btn_lab, 'LAB')
        
        ax_btn_combined = plt.axes([0.84, 0.02, 0.08, 0.025])
        btn_combined = Button(ax_btn_combined, 'Combined')
        
        ax_btn_waves = plt.axes([0.93, 0.02, 0.06, 0.025])
        btn_waves = Button(ax_btn_waves, 'Waves')
        
        def set_hsv(event):
            current['method'] = 'hsv'
            update_display()
        
        def set_lab(event):
            current['method'] = 'lab'
            update_display()
        
        def set_combined(event):
            current['method'] = 'combined'
            update_display()
        
        def toggle_waves(event):
            current['exclude_waves'] = not current['exclude_waves']
            btn_waves.label.set_text('Waves ✓' if current['exclude_waves'] else 'Waves ✗')
            update_display()
        
        btn_hsv.on_clicked(set_hsv)
        btn_lab.on_clicked(set_lab)
        btn_combined.on_clicked(set_combined)
        btn_waves.on_clicked(toggle_waves)
        
        # Initial display
        update_display()
        
        plt.show()


def analyze_single_frame(frame_number=248):
    """Quick analysis of a single frame"""
    
    segmenter = RGBOceanSegmenter()
    
    # Load data
    data = segmenter.load_rgb_thermal_pair(frame_number)
    
    # Segment using combined method
    masks = segmenter.segment_rgb_image(data['rgb_resized'], method='combined')
    
    # Apply to thermal
    ocean_thermal, land_thermal, stats = segmenter.apply_mask_to_thermal(
        data['thermal'], masks, exclude_waves=True
    )
    
    print(f"\nFrame {frame_number} Analysis:")
    print("=" * 40)
    print(f"Ocean coverage: {100*masks['ocean'].sum()/masks['ocean'].size:.1f}%")
    print(f"Land coverage: {100*masks['land'].sum()/masks['land'].size:.1f}%")
    print(f"Wave coverage: {100*masks['waves'].sum()/masks['waves'].size:.1f}%")
    print(f"\nOcean temp: {stats['ocean_mean']:.1f}°C (σ={stats['ocean_std']:.2f})")
    print(f"Land temp: {stats['land_mean']:.1f}°C (σ={stats['land_std']:.2f})")
    print(f"Wave zone temp: {stats['wave_mean']:.1f}°C (σ={stats['wave_std']:.2f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0,0].imshow(data['rgb_resized'])
    axes[0,0].set_title('RGB Image')
    axes[0,0].axis('off')
    
    mask_display = np.zeros((*masks['ocean'].shape, 3))
    mask_display[masks['ocean']] = [0, 0.3, 1]
    mask_display[masks['land']] = [0, 0.5, 0]
    mask_display[masks['waves']] = [1, 1, 1]
    
    axes[0,1].imshow(mask_display)
    axes[0,1].set_title('RGB Segmentation')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(data['thermal'], cmap='RdYlBu_r')
    axes[0,2].set_title('Original Thermal')
    axes[0,2].axis('off')
    
    axes[1,0].imshow(ocean_thermal, cmap='viridis')
    axes[1,0].set_title(f'Ocean Only\n{stats["ocean_mean"]:.1f}±{stats["ocean_std"]:.1f}°C')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(land_thermal, cmap='terrain')
    axes[1,1].set_title(f'Land Only\n{stats["land_mean"]:.1f}±{stats["land_std"]:.1f}°C')
    axes[1,1].axis('off')
    
    # Enhanced ocean
    ocean_enhanced = segmenter.enhance_ocean_visualization(ocean_thermal, masks)
    axes[1,2].imshow(ocean_enhanced, cmap='plasma')
    axes[1,2].set_title('Enhanced Ocean Thermal')
    axes[1,2].axis('off')
    
    plt.suptitle(f'RGB-Based Segmentation - Frame {frame_number}')
    plt.tight_layout()
    plt.show()


def main():
    """Main entry point"""
    print("RGB-Based Ocean Segmentation")
    print("=" * 40)
    
    print("\nOptions:")
    print("1. Interactive viewer")
    print("2. Single frame analysis")
    
    choice = input("\nChoice (1-2): ").strip()
    
    if choice == '1':
        viewer = RGBOceanViewer()
        print("\nLaunching interactive viewer...")
        print("Controls:")
        print("  - Frame slider: Navigate frames")
        print("  - Method buttons: HSV, LAB, or Combined segmentation")
        print("  - Waves button: Toggle wave exclusion")
        viewer.create_interactive_viewer()
    else:
        frame = int(input("Enter frame number (default 248): ") or "248")
        analyze_single_frame(frame)


if __name__ == "__main__":
    main()