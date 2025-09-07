#!/usr/bin/env python3
"""
Batch Ocean Processor
Process multiple thermal frames to extract and save ocean-only data
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import json
from ocean_thermal_analyzer import OceanThermalSegmenter
from tqdm import tqdm


class BatchOceanProcessor:
    """Batch process thermal frames to extract ocean data"""
    
    def __init__(self, base_path="data/100MEDIA", output_path="ocean_output"):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        self.segmenter = OceanThermalSegmenter()
        
    def process_all_frames(self, method='adaptive', save_images=True, save_data=True):
        """Process all available frames"""
        
        # Get all frames
        irg_files = sorted(self.base_path.glob("IRX_*.irg"))
        
        if not irg_files:
            tiff_files = sorted(self.base_path.glob("IRX_*.TIFF"))
            frame_numbers = [int(f.stem.split('_')[1]) for f in tiff_files]
        else:
            frame_numbers = [int(f.stem.split('_')[1]) for f in irg_files]
        
        print(f"Found {len(frame_numbers)} frames to process")
        
        # Results storage
        results = []
        
        # Process each frame
        for frame_num in tqdm(frame_numbers, desc="Processing frames"):
            try:
                result = self.process_single_frame(
                    frame_num, method, save_images, save_data
                )
                results.append(result)
            except Exception as e:
                print(f"\nError processing frame {frame_num}: {e}")
                continue
        
        # Save summary statistics
        self.save_summary(results)
        
        return results
    
    def process_single_frame(self, frame_number, method='adaptive', 
                           save_images=True, save_data=True):
        """Process a single frame"""
        
        # Load thermal data
        data = self.segmenter.load_thermal_frame(frame_number, self.base_path)
        temp_celsius = data['temp_celsius']
        
        # Segment ocean from land
        ocean_mask, land_mask = self.segmenter.segment_ocean_land(temp_celsius, method=method)
        
        # Get ocean statistics
        ocean_temps = temp_celsius[ocean_mask]
        land_temps = temp_celsius[land_mask]
        
        stats = {
            'frame_number': frame_number,
            'ocean_coverage_percent': 100 * ocean_mask.sum() / ocean_mask.size,
            'ocean_pixels': int(ocean_mask.sum()),
            'land_pixels': int(land_mask.sum()),
        }
        
        if len(ocean_temps) > 0:
            stats.update({
                'ocean_min': float(ocean_temps.min()),
                'ocean_max': float(ocean_temps.max()),
                'ocean_mean': float(ocean_temps.mean()),
                'ocean_std': float(ocean_temps.std()),
                'ocean_median': float(np.median(ocean_temps)),
            })
        
        if len(land_temps) > 0:
            stats.update({
                'land_min': float(land_temps.min()),
                'land_max': float(land_temps.max()),
                'land_mean': float(land_temps.mean()),
                'land_std': float(land_temps.std()),
            })
        
        # Save outputs
        if save_images:
            self.save_frame_images(frame_number, temp_celsius, ocean_mask, land_mask)
        
        if save_data:
            self.save_frame_data(frame_number, temp_celsius, ocean_mask, stats)
        
        return stats
    
    def save_frame_images(self, frame_number, temp_celsius, ocean_mask, land_mask):
        """Save visualization images for a frame"""
        
        # Create frame output directory
        frame_dir = self.output_path / f"frame_{frame_number:04d}"
        frame_dir.mkdir(exist_ok=True)
        
        # 1. Save ocean mask
        mask_img = Image.fromarray((ocean_mask * 255).astype(np.uint8))
        mask_img.save(frame_dir / "ocean_mask.png")
        
        # 2. Save ocean-only temperature (normalized)
        ocean_only = temp_celsius.copy()
        ocean_only[~ocean_mask] = np.nan
        
        valid_temps = ocean_only[ocean_mask]
        if len(valid_temps) > 0:
            ocean_min = valid_temps.min()
            ocean_max = valid_temps.max()
            
            # Normalize to 0-255 for saving
            ocean_normalized = (ocean_only - ocean_min) / (ocean_max - ocean_min + 1e-10)
            ocean_normalized[~ocean_mask] = 0
            ocean_img = Image.fromarray((ocean_normalized * 255).astype(np.uint8))
            ocean_img.save(frame_dir / "ocean_thermal_normalized.png")
            
            # Enhanced version with histogram equalization
            ocean_enhanced, _, _ = self.segmenter.enhance_ocean_thermal(temp_celsius, ocean_mask)
            ocean_enhanced[~ocean_mask] = 0
            enhanced_img = Image.fromarray((ocean_enhanced * 255).astype(np.uint8))
            enhanced_img.save(frame_dir / "ocean_thermal_enhanced.png")
        
        # 3. Create and save composite visualization
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original thermal
        im1 = axes[0].imshow(temp_celsius, cmap='RdYlBu_r')
        axes[0].set_title('Original Thermal')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # Segmentation
        mask_display = np.zeros((*ocean_mask.shape, 3))
        mask_display[ocean_mask] = [0, 0, 1]  # Blue for ocean
        mask_display[land_mask] = [0.5, 0.3, 0]  # Brown for land
        axes[1].imshow(mask_display)
        axes[1].set_title('Segmentation')
        axes[1].axis('off')
        
        # Ocean only
        if len(valid_temps) > 0:
            im2 = axes[2].imshow(ocean_only, cmap='viridis')
            axes[2].set_title(f'Ocean Only [{ocean_min:.1f}-{ocean_max:.1f}°C]')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046)
            
            # Enhanced
            im3 = axes[3].imshow(ocean_enhanced, cmap='plasma')
            axes[3].set_title('Enhanced Ocean')
            axes[3].axis('off')
            plt.colorbar(im3, ax=axes[3], fraction=0.046)
        
        plt.suptitle(f'Frame {frame_number:04d}')
        plt.tight_layout()
        plt.savefig(frame_dir / "composite.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_frame_data(self, frame_number, temp_celsius, ocean_mask, stats):
        """Save numerical data for a frame"""
        
        frame_dir = self.output_path / f"frame_{frame_number:04d}"
        frame_dir.mkdir(exist_ok=True)
        
        # Save statistics as JSON
        with open(frame_dir / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save ocean temperatures as numpy array
        ocean_temps = temp_celsius.copy()
        ocean_temps[~ocean_mask] = np.nan
        np.save(frame_dir / "ocean_temperatures.npy", ocean_temps)
        
        # Save mask
        np.save(frame_dir / "ocean_mask.npy", ocean_mask)
    
    def save_summary(self, results):
        """Save summary statistics for all frames"""
        
        if not results:
            return
        
        # Save all frame statistics
        with open(self.output_path / "all_frames_stats.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        frames = [r['frame_number'] for r in results]
        
        # Ocean coverage
        coverage = [r.get('ocean_coverage_percent', 0) for r in results]
        axes[0, 0].plot(frames, coverage, 'b-')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Ocean Coverage (%)')
        axes[0, 0].set_title('Ocean Coverage Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ocean temperature stats
        ocean_means = [r.get('ocean_mean', np.nan) for r in results]
        ocean_mins = [r.get('ocean_min', np.nan) for r in results]
        ocean_maxs = [r.get('ocean_max', np.nan) for r in results]
        
        axes[0, 1].plot(frames, ocean_means, 'b-', label='Mean')
        axes[0, 1].fill_between(frames, ocean_mins, ocean_maxs, alpha=0.3)
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].set_title('Ocean Temperature Range')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Ocean std deviation
        ocean_stds = [r.get('ocean_std', np.nan) for r in results]
        axes[0, 2].plot(frames, ocean_stds, 'g-')
        axes[0, 2].set_xlabel('Frame')
        axes[0, 2].set_ylabel('Std Dev (°C)')
        axes[0, 2].set_title('Ocean Temperature Variability')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Land temperature stats
        land_means = [r.get('land_mean', np.nan) for r in results]
        land_mins = [r.get('land_min', np.nan) for r in results]
        land_maxs = [r.get('land_max', np.nan) for r in results]
        
        axes[1, 0].plot(frames, land_means, 'r-', label='Mean')
        axes[1, 0].fill_between(frames, land_mins, land_maxs, alpha=0.3, color='red')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].set_title('Land Temperature Range')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Temperature difference (land - ocean)
        temp_diffs = [l - o for l, o in zip(land_means, ocean_means) 
                     if not np.isnan(l) and not np.isnan(o)]
        valid_frames = [f for f, l, o in zip(frames, land_means, ocean_means)
                       if not np.isnan(l) and not np.isnan(o)]
        
        if temp_diffs:
            axes[1, 1].plot(valid_frames, temp_diffs, 'purple')
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('Temperature Difference (°C)')
            axes[1, 1].set_title('Land - Ocean Temperature')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Overall statistics
        stats_text = "Overall Statistics:\n\n"
        stats_text += f"Frames processed: {len(results)}\n"
        stats_text += f"Avg ocean coverage: {np.mean(coverage):.1f}%\n\n"
        
        valid_ocean_means = [x for x in ocean_means if not np.isnan(x)]
        if valid_ocean_means:
            stats_text += f"Ocean temperature:\n"
            stats_text += f"  Mean: {np.mean(valid_ocean_means):.2f}°C\n"
            stats_text += f"  Range: {np.min(ocean_mins):.2f} - {np.max(ocean_maxs):.2f}°C\n\n"
        
        valid_land_means = [x for x in land_means if not np.isnan(x)]
        if valid_land_means:
            stats_text += f"Land temperature:\n"
            stats_text += f"  Mean: {np.mean(valid_land_means):.2f}°C\n"
            stats_text += f"  Range: {np.min(land_mins):.2f} - {np.max(land_maxs):.2f}°C\n"
        
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center')
        axes[1, 2].axis('off')
        
        plt.suptitle('Batch Processing Summary')
        plt.tight_layout()
        plt.savefig(self.output_path / "summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nSummary saved to {self.output_path}")
        print(stats_text)


def main():
    """Main entry point"""
    print("Batch Ocean Thermal Processor")
    print("=" * 40)
    
    processor = BatchOceanProcessor()
    
    print("\nProcessing options:")
    print("1. Process all frames")
    print("2. Process specific range")
    print("3. Quick test (first 5 frames)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        results = processor.process_all_frames()
    elif choice == '2':
        start = int(input("Start frame: "))
        end = int(input("End frame: "))
        # Limited implementation for specific range
        print(f"Processing frames {start} to {end}...")
        results = []
        for frame in range(start, end+1):
            try:
                result = processor.process_single_frame(frame)
                results.append(result)
            except:
                pass
        processor.save_summary(results)
    else:
        # Quick test
        print("Processing first 5 frames as test...")
        results = []
        for frame in range(1, 6):
            try:
                result = processor.process_single_frame(frame)
                results.append(result)
            except:
                pass
        processor.save_summary(results)
    
    print(f"\nProcessing complete!")
    print(f"Results saved to: {processor.output_path}")


if __name__ == "__main__":
    main()