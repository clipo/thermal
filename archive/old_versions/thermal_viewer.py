#!/usr/bin/env python3
"""
Autel 640T Thermal Data Viewer
Analyzes and visualizes thermal camera data from Autel 640T UAV
"""

import os
import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ThermalDataReader:
    """Read and parse thermal data from Autel 640T files"""
    
    def __init__(self, base_path="data"):
        self.base_path = Path(base_path)
        self.thermal_width = 640
        self.thermal_height = 512
        
    def read_irg_file(self, filepath):
        """
        Read .irg file (likely raw thermal data)
        IRG files appear to contain raw thermal sensor data
        """
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Try parsing as 16-bit unsigned integers (common for thermal data)
            # File size suggests 640x512 resolution with additional metadata
            expected_pixels = self.thermal_width * self.thermal_height
            
            # Try different interpretations
            results = {}
            
            # Method 1: Direct 16-bit values
            if len(data) >= expected_pixels * 2:
                thermal_data = np.frombuffer(data[:expected_pixels*2], dtype=np.uint16)
                thermal_data = thermal_data.reshape((self.thermal_height, self.thermal_width))
                results['uint16'] = thermal_data
            
            # Method 2: Check for header and parse accordingly
            # Many thermal formats have headers with calibration data
            header_size = len(data) - (expected_pixels * 2)
            if header_size > 0:
                # Try reading with header offset
                thermal_data = np.frombuffer(data[header_size:], dtype=np.uint16)
                if len(thermal_data) == expected_pixels:
                    thermal_data = thermal_data.reshape((self.thermal_height, self.thermal_width))
                    results['uint16_with_header'] = thermal_data
                    
                # Also extract potential header info
                results['header_size'] = header_size
                results['header_data'] = data[:header_size]
            
            # Method 3: Try as float32 (some formats use floating point)
            if len(data) >= expected_pixels * 4:
                thermal_data = np.frombuffer(data[:expected_pixels*4], dtype=np.float32)
                thermal_data = thermal_data.reshape((self.thermal_height, self.thermal_width))
                results['float32'] = thermal_data
            
            return results
            
        except Exception as e:
            print(f"Error reading IRG file: {e}")
            return None
    
    def read_thermal_jpg(self, filepath):
        """Read thermal JPG file"""
        try:
            img = Image.open(filepath)
            img_array = np.array(img)
            
            # Extract EXIF data if available (might contain temperature calibration)
            exif_data = img._getexif() if hasattr(img, '_getexif') else None
            
            return {
                'image': img_array,
                'mode': img.mode,
                'size': img.size,
                'exif': exif_data
            }
        except Exception as e:
            print(f"Error reading JPG file: {e}")
            return None
    
    def read_thermal_tiff(self, filepath):
        """Read thermal TIFF file - often contains raw temperature data"""
        try:
            img = Image.open(filepath)
            img_array = np.array(img)
            
            # TIFF files often store actual temperature values
            # Check if it's 16-bit (common for thermal data)
            if img.mode == 'I;16':
                # 16-bit integer mode
                img_array = np.array(img, dtype=np.uint16)
            
            return {
                'image': img_array,
                'mode': img.mode,
                'size': img.size,
                'info': img.info
            }
        except Exception as e:
            print(f"Error reading TIFF file: {e}")
            return None
    
    def convert_raw_to_temperature(self, raw_values, emissivity=0.95):
        """
        Convert raw thermal values to temperature
        This is a simplified conversion - actual formula depends on camera calibration
        """
        # Typical conversion for uncooled microbolometer sensors
        # These constants would normally come from camera calibration
        R = 16384.0  # Planck R1 constant
        B = 1428.0   # Planck B constant  
        F = 1.0      # Planck F constant
        O = -7340.0  # Planck O constant
        
        # Convert raw values to radiance
        raw_float = raw_values.astype(np.float64)
        
        # Prevent division by zero
        raw_float[raw_float == 0] = 1
        
        # Stefan-Boltzmann law approximation
        # Temperature in Kelvin
        temp_kelvin = B / np.log(R / (raw_float - O) + F)
        
        # Convert to Celsius
        temp_celsius = temp_kelvin - 273.15
        
        # Apply emissivity correction
        temp_celsius = temp_celsius * emissivity
        
        return temp_celsius
    
    def analyze_color_mapping(self, thermal_jpg, raw_values):
        """Analyze how JPG colors map to temperature values"""
        
        # Convert JPG to grayscale if needed
        if len(thermal_jpg.shape) == 3:
            # Convert RGB to grayscale using standard formula
            gray_jpg = np.dot(thermal_jpg[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray_jpg = thermal_jpg
        
        # Normalize both to 0-255 range for comparison
        norm_jpg = gray_jpg.astype(np.float32)
        
        # Normalize raw values
        raw_min, raw_max = np.min(raw_values), np.max(raw_values)
        norm_raw = ((raw_values - raw_min) / (raw_max - raw_min) * 255).astype(np.float32)
        
        # Calculate correlation
        correlation = np.corrcoef(norm_jpg.flatten(), norm_raw.flatten())[0, 1]
        
        # Find mapping function
        # Sample points for mapping analysis
        sample_indices = np.random.choice(norm_jpg.size, min(1000, norm_jpg.size), replace=False)
        jpg_samples = norm_jpg.flatten()[sample_indices]
        raw_samples = norm_raw.flatten()[sample_indices]
        
        # Fit polynomial to understand mapping
        coeffs = np.polyfit(jpg_samples, raw_samples, 2)
        
        return {
            'correlation': correlation,
            'mapping_coeffs': coeffs,
            'jpg_range': (np.min(norm_jpg), np.max(norm_jpg)),
            'raw_range': (raw_min, raw_max),
            'normalized_raw': norm_raw
        }


class ThermalViewer:
    """Interactive viewer for thermal data analysis"""
    
    def __init__(self, data_path="data"):
        self.reader = ThermalDataReader(data_path)
        self.data_path = Path(data_path)
        self.current_frame = None
        self.frames = []
        
    def load_frame_set(self, frame_number):
        """Load all files for a given frame number"""
        frame_data = {}
        
        # Find the appropriate directory
        media_dir = self.data_path / "100MEDIA"
        
        # Load IRG file
        irg_path = media_dir / f"IRX_{frame_number:04d}.irg"
        if irg_path.exists():
            frame_data['irg'] = self.reader.read_irg_file(irg_path)
        
        # Load thermal JPG
        jpg_path = media_dir / f"IRX_{frame_number:04d}.jpg"
        if jpg_path.exists():
            frame_data['jpg'] = self.reader.read_thermal_jpg(jpg_path)
        
        # Load TIFF
        tiff_path = media_dir / f"IRX_{frame_number:04d}.TIFF"
        if tiff_path.exists():
            frame_data['tiff'] = self.reader.read_thermal_tiff(tiff_path)
        
        # Load RGB image
        rgb_path = media_dir / f"MAX_{frame_number:04d}.JPG"
        if rgb_path.exists():
            frame_data['rgb'] = Image.open(rgb_path)
        
        frame_data['frame_number'] = frame_number
        return frame_data
    
    def visualize_frame(self, frame_data):
        """Create comprehensive visualization of thermal data"""
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Frame {frame_data["frame_number"]:04d} - Thermal Data Analysis', fontsize=16)
        
        # Row 1: Original files
        # Thermal JPG
        if 'jpg' in frame_data and frame_data['jpg']:
            axes[0, 0].imshow(frame_data['jpg']['image'], cmap='hot')
            axes[0, 0].set_title(f'Thermal JPG\n{frame_data["jpg"]["mode"]} mode')
            axes[0, 0].axis('off')
        
        # TIFF data
        if 'tiff' in frame_data and frame_data['tiff']:
            tiff_img = frame_data['tiff']['image']
            im1 = axes[0, 1].imshow(tiff_img, cmap='hot')
            axes[0, 1].set_title(f'TIFF Raw\nShape: {tiff_img.shape}, Range: [{tiff_img.min():.0f}, {tiff_img.max():.0f}]')
            axes[0, 1].axis('off')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # IRG data (raw thermal)
        if 'irg' in frame_data and frame_data['irg']:
            # Try different interpretations
            if 'uint16' in frame_data['irg']:
                raw_data = frame_data['irg']['uint16']
                im2 = axes[0, 2].imshow(raw_data, cmap='hot')
                axes[0, 2].set_title(f'IRG Raw (uint16)\nRange: [{raw_data.min():.0f}, {raw_data.max():.0f}]')
                axes[0, 2].axis('off')
                plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
                
                # Convert to temperature
                temp_data = self.reader.convert_raw_to_temperature(raw_data)
                im3 = axes[0, 3].imshow(temp_data, cmap='hot')
                axes[0, 3].set_title(f'Temperature (°C)\nRange: [{temp_data.min():.1f}, {temp_data.max():.1f}]')
                axes[0, 3].axis('off')
                plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, label='°C')
        
        # RGB image
        if 'rgb' in frame_data:
            axes[1, 0].imshow(frame_data['rgb'])
            axes[1, 0].set_title('RGB Image')
            axes[1, 0].axis('off')
        
        # Row 2: Analysis
        # Histogram comparison
        if 'jpg' in frame_data and 'irg' in frame_data:
            jpg_img = frame_data['jpg']['image']
            if len(jpg_img.shape) == 3:
                jpg_gray = np.dot(jpg_img[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                jpg_gray = jpg_img
            
            axes[1, 1].hist(jpg_gray.flatten(), bins=50, alpha=0.5, label='JPG', color='blue')
            
            if 'uint16' in frame_data['irg']:
                raw_data = frame_data['irg']['uint16']
                axes[1, 1].hist(raw_data.flatten(), bins=50, alpha=0.5, label='IRG Raw', color='red')
            
            axes[1, 1].set_title('Histogram Comparison')
            axes[1, 1].set_xlabel('Pixel Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        # Difference analysis
        if 'jpg' in frame_data and 'irg' in frame_data and 'uint16' in frame_data['irg']:
            jpg_img = frame_data['jpg']['image']
            raw_data = frame_data['irg']['uint16']
            
            # Analyze mapping
            mapping = self.reader.analyze_color_mapping(jpg_img, raw_data)
            
            # Show normalized comparison
            axes[1, 2].imshow(mapping['normalized_raw'], cmap='hot')
            axes[1, 2].set_title(f'Normalized Raw\nCorrelation: {mapping["correlation"]:.3f}')
            axes[1, 2].axis('off')
            
            # Scatter plot of mapping
            if len(jpg_img.shape) == 3:
                jpg_gray = np.dot(jpg_img[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                jpg_gray = jpg_img
            
            # Sample for scatter plot
            sample_size = min(5000, jpg_gray.size)
            idx = np.random.choice(jpg_gray.size, sample_size, replace=False)
            
            axes[1, 3].scatter(jpg_gray.flatten()[idx], 
                             mapping['normalized_raw'].flatten()[idx],
                             alpha=0.3, s=1)
            axes[1, 3].set_xlabel('JPG Pixel Value')
            axes[1, 3].set_ylabel('Normalized Raw Value')
            axes[1, 3].set_title('JPG vs Raw Correlation')
            axes[1, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_viewer(self):
        """Create an interactive viewer with frame navigation"""
        
        # Get available frames
        media_dir = self.data_path / "100MEDIA"
        irg_files = list(media_dir.glob("IRX_*.irg"))
        frame_numbers = sorted([int(f.stem.split('_')[1]) for f in irg_files])
        
        if not frame_numbers:
            print("No IRG files found!")
            return
        
        print(f"Found {len(frame_numbers)} frames: {frame_numbers[0]} to {frame_numbers[-1]}")
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Initial frame
        current_idx = [0]
        frame_data = self.load_frame_set(frame_numbers[current_idx[0]])
        
        # Create subplots
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Define axes
        ax_jpg = fig.add_subplot(gs[0, 0])
        ax_tiff = fig.add_subplot(gs[0, 1])
        ax_irg = fig.add_subplot(gs[0, 2])
        ax_temp = fig.add_subplot(gs[0, 3])
        ax_rgb = fig.add_subplot(gs[1, 0])
        ax_hist = fig.add_subplot(gs[1, 1])
        ax_norm = fig.add_subplot(gs[1, 2])
        ax_scatter = fig.add_subplot(gs[1, 3])
        ax_profile = fig.add_subplot(gs[2, :2])
        ax_stats = fig.add_subplot(gs[2, 2:])
        
        def update_display(val=None):
            """Update all displays"""
            if val is not None:
                current_idx[0] = int(val)
            
            frame_num = frame_numbers[current_idx[0]]
            frame_data = self.load_frame_set(frame_num)
            
            # Clear all axes
            for ax in [ax_jpg, ax_tiff, ax_irg, ax_temp, ax_rgb, ax_hist, 
                      ax_norm, ax_scatter, ax_profile, ax_stats]:
                ax.clear()
            
            fig.suptitle(f'Frame {frame_num:04d} - Thermal Data Analysis', fontsize=16)
            
            # Display images
            if 'jpg' in frame_data and frame_data['jpg']:
                ax_jpg.imshow(frame_data['jpg']['image'], cmap='hot')
                ax_jpg.set_title('Thermal JPG')
                ax_jpg.axis('off')
            
            if 'tiff' in frame_data and frame_data['tiff']:
                tiff_img = frame_data['tiff']['image']
                im = ax_tiff.imshow(tiff_img, cmap='hot')
                ax_tiff.set_title(f'TIFF [{tiff_img.min():.0f}, {tiff_img.max():.0f}]')
                ax_tiff.axis('off')
            
            if 'irg' in frame_data and frame_data['irg'] and 'uint16' in frame_data['irg']:
                raw_data = frame_data['irg']['uint16']
                im = ax_irg.imshow(raw_data, cmap='hot')
                ax_irg.set_title(f'IRG Raw [{raw_data.min():.0f}, {raw_data.max():.0f}]')
                ax_irg.axis('off')
                
                # Temperature
                temp_data = self.reader.convert_raw_to_temperature(raw_data)
                im = ax_temp.imshow(temp_data, cmap='hot')
                ax_temp.set_title(f'Temp (°C) [{temp_data.min():.1f}, {temp_data.max():.1f}]')
                ax_temp.axis('off')
                
                # Line profile through center
                center_y = raw_data.shape[0] // 2
                ax_profile.plot(raw_data[center_y, :], label='Raw Values', alpha=0.7)
                ax_profile.plot(temp_data[center_y, :], label='Temperature (°C)', alpha=0.7)
                ax_profile.set_xlabel('Pixel Position')
                ax_profile.set_ylabel('Value')
                ax_profile.set_title('Horizontal Profile (Center Line)')
                ax_profile.legend()
                ax_profile.grid(True, alpha=0.3)
                
                # Statistics
                stats_text = f"Raw Data Statistics:\n"
                stats_text += f"Min: {raw_data.min():.0f}\n"
                stats_text += f"Max: {raw_data.max():.0f}\n"
                stats_text += f"Mean: {raw_data.mean():.0f}\n"
                stats_text += f"Std: {raw_data.std():.0f}\n\n"
                stats_text += f"Temperature Statistics:\n"
                stats_text += f"Min: {temp_data.min():.1f}°C\n"
                stats_text += f"Max: {temp_data.max():.1f}°C\n"
                stats_text += f"Mean: {temp_data.mean():.1f}°C\n"
                stats_text += f"Std: {temp_data.std():.1f}°C"
                
                ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                            fontsize=10, verticalalignment='center')
                ax_stats.axis('off')
            
            if 'rgb' in frame_data:
                ax_rgb.imshow(frame_data['rgb'])
                ax_rgb.set_title('RGB Image')
                ax_rgb.axis('off')
            
            # Analysis plots
            if 'jpg' in frame_data and 'irg' in frame_data and frame_data['jpg'] and frame_data['irg']:
                jpg_img = frame_data['jpg']['image']
                if len(jpg_img.shape) == 3:
                    jpg_gray = np.dot(jpg_img[...,:3], [0.2989, 0.5870, 0.1140])
                else:
                    jpg_gray = jpg_img
                
                ax_hist.hist(jpg_gray.flatten(), bins=50, alpha=0.5, label='JPG', color='blue')
                
                if 'uint16' in frame_data['irg']:
                    raw_data = frame_data['irg']['uint16']
                    raw_norm = (raw_data - raw_data.min()) / (raw_data.max() - raw_data.min()) * 255
                    ax_hist.hist(raw_norm.flatten(), bins=50, alpha=0.5, label='IRG (normalized)', color='red')
                    
                    # Mapping analysis
                    mapping = self.reader.analyze_color_mapping(jpg_img, raw_data)
                    ax_norm.imshow(mapping['normalized_raw'], cmap='hot')
                    ax_norm.set_title(f'Normalized Raw (r={mapping["correlation"]:.3f})')
                    ax_norm.axis('off')
                    
                    # Scatter plot
                    sample_size = min(2000, jpg_gray.size)
                    idx = np.random.choice(jpg_gray.size, sample_size, replace=False)
                    ax_scatter.scatter(jpg_gray.flatten()[idx], 
                                     mapping['normalized_raw'].flatten()[idx],
                                     alpha=0.3, s=1)
                    ax_scatter.set_xlabel('JPG Value')
                    ax_scatter.set_ylabel('Normalized Raw')
                    ax_scatter.set_title('Correlation Plot')
                    ax_scatter.grid(True, alpha=0.3)
                
                ax_hist.set_xlabel('Pixel Value')
                ax_hist.set_ylabel('Frequency')
                ax_hist.legend()
            
            fig.canvas.draw_idle()
        
        # Add slider for frame navigation
        ax_slider = plt.axes([0.1, 0.02, 0.8, 0.02])
        slider = Slider(ax_slider, 'Frame', 0, len(frame_numbers)-1, 
                       valinit=0, valstep=1)
        slider.on_changed(update_display)
        
        # Initial display
        update_display()
        
        plt.show()
        
        return fig, slider


def main():
    """Main entry point"""
    print("Autel 640T Thermal Data Viewer")
    print("=" * 40)
    
    # Create viewer
    viewer = ThermalViewer()
    
    # Create interactive viewer
    print("\nLaunching interactive viewer...")
    print("Use the slider to navigate between frames")
    print("Close the window to exit")
    
    viewer.create_interactive_viewer()


if __name__ == "__main__":
    main()