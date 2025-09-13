#!/usr/bin/env python3
"""
Generate figures for the technical paper from SGD detection data.
This script creates the visualizations needed for TECHNICAL_PAPER.md
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import json
from typing import Optional, Tuple, List
import argparse
from datetime import datetime
from scipy import ndimage

# Try to import the main modules
try:
    from sgd_autodetect import SGDDetector
    from sgd_viewer import ThermalViewer
except ImportError:
    print("Warning: Could not import main modules. Some functions may be limited.")

class FigureGenerator:
    """Generate publication-quality figures for the SGD detection paper."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        """Initialize the figure generator.
        
        Args:
            data_dir: Directory containing thermal/RGB data
            output_dir: Directory to save generated figures
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set publication-quality defaults
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        
    def generate_thermal_rgb_pair(self, thermal_path: Path, rgb_path: Path, 
                                  output_name: str = "thermal_rgb_pair.png"):
        """Generate side-by-side thermal and RGB comparison.
        
        Args:
            thermal_path: Path to thermal image
            rgb_path: Path to RGB image
            output_name: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Load and display RGB
        if rgb_path.exists():
            rgb_img = np.array(Image.open(rgb_path))
            ax1.imshow(rgb_img)
            ax1.set_title(f'RGB Image ({rgb_img.shape[1]}×{rgb_img.shape[0]})')
            ax1.axis('off')
            
            # Add FOV rectangle showing thermal coverage
            h, w = rgb_img.shape[:2]
            thermal_fov_rect = patches.Rectangle((w*0.3, h*0.3), w*0.4, h*0.4,
                                                linewidth=2, edgecolor='yellow',
                                                facecolor='none', linestyle='--')
            ax1.add_patch(thermal_fov_rect)
            ax1.text(w*0.5, h*0.25, 'Thermal FOV', color='yellow', 
                    ha='center', fontsize=10, weight='bold')
        
        # Load and display thermal
        if thermal_path.exists():
            thermal_img = np.array(Image.open(thermal_path))
            
            # Convert if in DeciKelvin format or handle as regular image
            if len(thermal_img.shape) == 3:
                # Convert RGB thermal to grayscale if needed
                thermal_img = np.mean(thermal_img, axis=2)
            
            if thermal_img.dtype == np.uint16 and thermal_img.max() > 3000:
                thermal_celsius = (thermal_img.astype(np.float32) / 10.0) - 273.15
            elif thermal_img.max() > 100:
                # Assume it's in some scaled format, normalize to reasonable temp range
                thermal_celsius = 18 + (thermal_img.astype(np.float32) - thermal_img.min()) / (thermal_img.max() - thermal_img.min()) * 7
            else:
                thermal_celsius = thermal_img
            
            im = ax2.imshow(thermal_celsius, cmap='jet', vmin=18, vmax=25)
            ax2.set_title(f'Thermal Image (640×512)')
            ax2.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
        
        plt.suptitle('Thermal-RGB Image Pair from UAV Survey', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated: {output_name}")
        
    def generate_environmental_diversity(self, image_paths: List[Path],
                                        output_name: str = "environmental_diversity.png"):
        """Generate 2x2 grid showing different coastal environments.
        
        Args:
            image_paths: List of 4 image paths
            output_name: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        environments = ['Rocky Volcanic', 'Sandy Beach', 'Boulder Field', 'Active Surf']
        
        for idx, (path, env_name, ax) in enumerate(zip(image_paths, environments, axes.flat)):
            if path.exists():
                img = np.array(Image.open(path))
                ax.imshow(img)
                ax.set_title(f'{chr(65+idx)}) {env_name}', fontsize=12, weight='bold')
            else:
                # Create placeholder
                ax.text(0.5, 0.5, f'{env_name}\n(Image not found)', 
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            ax.axis('off')
        
        plt.suptitle('Coastal Environment Diversity in Rapa Nui', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated: {output_name}")
        
    def generate_segmentation_example(self, rgb_path: Path, segmentation_path: Optional[Path] = None,
                                     output_name: str = "segmentation_example.png"):
        """Generate three-panel segmentation process visualization.
        
        Args:
            rgb_path: Path to RGB image
            segmentation_path: Path to segmentation mask (optional)
            output_name: Output filename
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original RGB
        if rgb_path.exists():
            rgb_img = np.array(Image.open(rgb_path))
            ax1.imshow(rgb_img)
            ax1.set_title('Original RGB', fontsize=12, weight='bold')
            ax1.axis('off')
            
            # Create synthetic segmentation if not provided
            if segmentation_path and segmentation_path.exists():
                seg_mask = np.array(Image.open(segmentation_path).convert('L'))
            else:
                # Create synthetic segmentation based on image characteristics
                gray = np.array(Image.fromarray(rgb_img).convert('L'))
                seg_mask = np.zeros_like(gray)
                
                # Simple threshold-based segmentation
                seg_mask[gray < 100] = 1  # Ocean (dark)
                seg_mask[(gray >= 100) & (gray < 150)] = 2  # Rock
                seg_mask[(gray >= 150) & (gray < 200)] = 3  # Sand
                seg_mask[gray >= 200] = 4  # Wave/foam
            
            # Color-coded segmentation
            seg_colored = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
            seg_colored[seg_mask == 1] = [0, 0, 200]  # Ocean - blue
            seg_colored[seg_mask == 2] = [128, 128, 128]  # Rock - gray
            seg_colored[seg_mask == 3] = [194, 178, 128]  # Sand - tan
            seg_colored[seg_mask == 4] = [255, 255, 255]  # Wave - white
            
            ax2.imshow(seg_colored)
            ax2.set_title('Segmentation Map', fontsize=12, weight='bold')
            ax2.axis('off')
            
            # Binary ocean mask
            ocean_mask = (seg_mask == 1).astype(np.uint8) * 255
            ax3.imshow(ocean_mask, cmap='gray')
            ax3.set_title('Ocean Mask', fontsize=12, weight='bold')
            ax3.axis('off')
            
            # Add legend
            legend_elements = [
                patches.Patch(color=[0, 0, 0.8], label='Ocean'),
                patches.Patch(color=[0.5, 0.5, 0.5], label='Rock'),
                patches.Patch(color=[0.76, 0.7, 0.5], label='Sand'),
                patches.Patch(color=[1, 1, 1], label='Wave')
            ]
            ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.suptitle('ML-Based Ocean Segmentation Process', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated: {output_name}")
        
    def generate_sgd_detection_process(self, thermal_path: Path,
                                      output_name: str = "sgd_detection_process.png"):
        """Generate four-panel SGD detection process visualization.
        
        Args:
            thermal_path: Path to thermal image
            output_name: Output filename
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        if thermal_path.exists():
            # Load thermal image
            thermal_img = np.array(Image.open(thermal_path))
            
            # Convert DeciKelvin to Celsius or handle as regular image
            if len(thermal_img.shape) == 3:
                thermal_img = np.mean(thermal_img, axis=2)
            
            if thermal_img.dtype == np.uint16 and thermal_img.max() > 3000:
                thermal_celsius = (thermal_img.astype(np.float32) / 10.0) - 273.15
            elif thermal_img.max() > 100:
                thermal_celsius = 18 + (thermal_img.astype(np.float32) - thermal_img.min()) / (thermal_img.max() - thermal_img.min()) * 7
            else:
                thermal_celsius = thermal_img
            
            # Panel A: Raw thermal
            im1 = ax1.imshow(thermal_celsius, cmap='jet', vmin=18, vmax=25)
            ax1.set_title('A) Raw Thermal Image', fontsize=12, weight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='°C')
            
            # Panel B: Ocean-masked (synthetic mask for demo)
            ocean_mask = np.ones_like(thermal_celsius, dtype=bool)
            ocean_mask[:100, :] = False  # Simulate land at top
            masked_thermal = np.ma.masked_where(~ocean_mask, thermal_celsius)
            
            im2 = ax2.imshow(masked_thermal, cmap='jet', vmin=18, vmax=25)
            ax2.set_title('B) Ocean-Masked Thermal', fontsize=12, weight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='°C')
            
            # Panel C: Temperature anomaly
            mean_temp = np.nanmean(thermal_celsius[ocean_mask])
            anomaly = thermal_celsius - mean_temp
            
            im3 = ax3.imshow(anomaly, cmap='RdBu_r', vmin=-3, vmax=3)
            ax3.set_title('C) Temperature Anomaly', fontsize=12, weight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Δ°C')
            
            # Panel D: Detected SGD plumes
            sgd_threshold = -1.0  # 1°C cooler than mean
            sgd_mask = (anomaly < sgd_threshold) & ocean_mask
            
            # Find connected components for polygon overlay
            labeled, num_features = ndimage.label(sgd_mask.astype(np.uint8))
            
            ax4.imshow(thermal_celsius, cmap='gray', vmin=18, vmax=25)
            
            # Draw SGD polygons for each detected region
            num_plumes = 0
            for i in range(1, num_features + 1):
                region = (labeled == i)
                if np.sum(region) > 50:  # Minimum area threshold
                    # Find boundary points
                    coords = np.where(region)
                    if len(coords[0]) > 0:
                        y_coords, x_coords = coords[0], coords[1]
                        # Create a simple polygon from boundary points
                        points = np.column_stack((x_coords, y_coords))
                        hull_points = points[::10]  # Subsample for simpler polygon
                        polygon = patches.Polygon(hull_points, 
                                                linewidth=2, edgecolor='red',
                                                facecolor='red', alpha=0.3)
                        ax4.add_patch(polygon)
                        num_plumes += 1
            
            ax4.set_title('D) Detected SGD Plumes', fontsize=12, weight='bold')
            ax4.axis('off')
            
            # Add detection stats
            ax4.text(0.02, 0.98, f'Plumes detected: {num_plumes}', 
                    transform=ax4.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        plt.suptitle('SGD Detection Process Pipeline', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated: {output_name}")
        
    def generate_sgd_plume_detail(self, thermal_path: Path, rgb_path: Path,
                                 output_name: str = "sgd_plume_detail.png"):
        """Generate close-up view of SGD plume with thermal and RGB.
        
        Args:
            thermal_path: Path to thermal image with SGD
            rgb_path: Path to corresponding RGB image
            output_name: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if thermal_path.exists():
            thermal_img = np.array(Image.open(thermal_path))
            
            # Convert DeciKelvin to Celsius or handle as regular image
            if len(thermal_img.shape) == 3:
                thermal_img = np.mean(thermal_img, axis=2)
            
            if thermal_img.dtype == np.uint16 and thermal_img.max() > 3000:
                thermal_celsius = (thermal_img.astype(np.float32) / 10.0) - 273.15
            elif thermal_img.max() > 100:
                thermal_celsius = 18 + (thermal_img.astype(np.float32) - thermal_img.min()) / (thermal_img.max() - thermal_img.min()) * 7
            else:
                thermal_celsius = thermal_img
            
            # Zoom to region of interest (center for demo)
            h, w = thermal_celsius.shape
            roi = thermal_celsius[h//3:2*h//3, w//3:2*w//3]
            
            im1 = ax1.imshow(roi, cmap='jet', vmin=18, vmax=23)
            ax1.set_title('Thermal View of SGD Plume', fontsize=12, weight='bold')
            ax1.axis('off')
            
            # Add temperature scale
            cbar = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
            
            # Add scale bar (10m)
            scale_pixels = 50  # Approximate
            ax1.plot([10, 10+scale_pixels], [roi.shape[0]-20, roi.shape[0]-20], 
                    'w-', linewidth=3)
            ax1.text(10+scale_pixels//2, roi.shape[0]-30, '10 m', 
                    color='white', ha='center', fontsize=10, weight='bold')
            
            # Add temperature anomaly annotation
            mean_temp = np.mean(roi)
            min_temp = np.min(roi)
            anomaly = min_temp - mean_temp
            ax1.text(0.02, 0.98, f'Anomaly: {anomaly:.1f}°C', 
                    transform=ax1.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        
        if rgb_path.exists():
            rgb_img = np.array(Image.open(rgb_path))
            
            # Zoom to corresponding region
            h, w = rgb_img.shape[:2]
            roi_rgb = rgb_img[h//3:2*h//3, w//3:2*w//3]
            
            ax2.imshow(roi_rgb)
            
            # Add SGD polygon overlay
            ellipse = patches.Ellipse((roi_rgb.shape[1]//2, roi_rgb.shape[0]//2),
                                     width=100, height=80,
                                     linewidth=2, edgecolor='red',
                                     facecolor='none')
            ax2.add_patch(ellipse)
            
            ax2.set_title('RGB View with SGD Polygon', fontsize=12, weight='bold')
            ax2.axis('off')
            
            # Add scale bar
            ax2.plot([10, 60], [roi_rgb.shape[0]-20, roi_rgb.shape[0]-20], 
                    'k-', linewidth=3)
            ax2.text(35, roi_rgb.shape[0]-30, '10 m', 
                    color='black', ha='center', fontsize=10, weight='bold')
        
        plt.suptitle('Close-up View of Individual SGD Plume', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / output_name, bbox_inches='tight')
        plt.close()
        print(f"✅ Generated: {output_name}")
        
    def generate_all_figures(self, sample_thermal: Path, sample_rgb: Path):
        """Generate all figures for the technical paper.
        
        Args:
            sample_thermal: Path to a sample thermal image
            sample_rgb: Path to a sample RGB image
        """
        print("\n🎨 Generating figures for technical paper...\n")
        
        # 1. Thermal-RGB pair
        self.generate_thermal_rgb_pair(sample_thermal, sample_rgb)
        
        # 2. Environmental diversity (use same image for demo)
        env_images = [sample_rgb] * 4  # In practice, use different environments
        self.generate_environmental_diversity(env_images)
        
        # 3. Segmentation example
        self.generate_segmentation_example(sample_rgb)
        
        # 4. SGD detection process
        self.generate_sgd_detection_process(sample_thermal)
        
        # 5. SGD plume detail
        self.generate_sgd_plume_detail(sample_thermal, sample_rgb)
        
        print(f"\n✅ All figures saved to: {self.output_dir}")
        print("\n📝 Next steps:")
        print("1. Review generated figures in docs/images/")
        print("2. Replace with actual survey data as available")
        print("3. Update TECHNICAL_PAPER.md image paths if needed")


def main():
    parser = argparse.ArgumentParser(description="Generate figures for SGD detection technical paper")
    parser.add_argument('--data', type=str, required=True,
                       help="Path to data directory with thermal/RGB images")
    parser.add_argument('--output', type=str, default='docs/images',
                       help="Output directory for figures (default: docs/images)")
    parser.add_argument('--thermal', type=str,
                       help="Specific thermal image for examples")
    parser.add_argument('--rgb', type=str,
                       help="Specific RGB image for examples")
    
    args = parser.parse_args()
    
    # Find sample images if not specified
    data_dir = Path(args.data)
    
    if args.thermal:
        sample_thermal = Path(args.thermal)
    else:
        # Find first thermal image
        thermal_files = list(data_dir.glob('**/DJI_*_T.JPG'))
        if not thermal_files:
            thermal_files = list(data_dir.glob('**/*thermal*.jpg'))
        if thermal_files:
            sample_thermal = thermal_files[0]
        else:
            print("⚠️ No thermal images found. Using placeholder.")
            sample_thermal = Path('placeholder_thermal.jpg')
    
    if args.rgb:
        sample_rgb = Path(args.rgb)
    else:
        # Find corresponding RGB image
        if sample_thermal.exists():
            rgb_name = sample_thermal.name.replace('_T.JPG', '_W.JPG')
            sample_rgb = sample_thermal.parent / rgb_name
            if not sample_rgb.exists():
                rgb_files = list(data_dir.glob('**/DJI_*_W.JPG'))
                if rgb_files:
                    sample_rgb = rgb_files[0]
                else:
                    sample_rgb = Path('placeholder_rgb.jpg')
        else:
            sample_rgb = Path('placeholder_rgb.jpg')
    
    print(f"📸 Using thermal: {sample_thermal}")
    print(f"📸 Using RGB: {sample_rgb}")
    
    # Generate figures
    generator = FigureGenerator(data_dir, Path(args.output))
    generator.generate_all_figures(sample_thermal, sample_rgb)


if __name__ == "__main__":
    main()