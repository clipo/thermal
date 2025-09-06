#!/usr/bin/env python3
"""
Integrated SGD Detector with Built-in RGB-Thermal Alignment
Submarine Groundwater Discharge detection with automatic image alignment
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pathlib import Path
from scipy import ndimage
from skimage import morphology, measure, color
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import ML segmentation module
try:
    from ml_segmentation_fast import FastMLSegmenter
    ML_AVAILABLE = True
    print("Fast ML segmentation module loaded")
except ImportError:
    try:
        from ml_segmentation import MLSegmenter as FastMLSegmenter
        ML_AVAILABLE = True
        print("Standard ML segmentation module loaded")
    except ImportError:
        ML_AVAILABLE = False
        print("ML segmentation not available. Using rule-based segmentation.")


class IntegratedSGDDetector:
    """SGD Detector with built-in RGB-thermal alignment"""
    
    def __init__(self, temp_threshold=1.0, min_area=50, base_path="data/100MEDIA", 
                 use_ml=True, ml_model_path="segmentation_model.pkl"):
        """
        Initialize integrated SGD detector
        
        Parameters:
        - temp_threshold: Temperature difference (°C) below ocean mean for SGD
        - min_area: Minimum area (pixels) for valid SGD plume
        - base_path: Path to data directory
        - use_ml: Use ML segmentation if available (default True)
        - ml_model_path: Path to ML model file (default "segmentation_model.pkl")
        """
        self.temp_threshold = temp_threshold
        self.min_area = min_area
        self.base_path = Path(base_path)
        self.use_ml = use_ml
        self.ml_model_path = ml_model_path
        
        # Initialize ML segmenter if requested and available
        self.ml_segmenter = None
        if self.use_ml and ML_AVAILABLE:
            self.ml_segmenter = FastMLSegmenter(model_path=ml_model_path)
            if self.ml_segmenter.classifier is not None:
                print(f"Using fast ML-based segmentation with model: {ml_model_path}")
            else:
                print(f"ML model not found at {ml_model_path}. Using rule-based segmentation.")
                print("Train a model using segmentation_trainer.py for better results.")
                self.ml_segmenter = None
        
        # Fixed alignment parameters for Autel 640T
        # RGB: 4096x3072, Thermal: 640x512
        # IMPORTANT: Thermal has narrower FOV (~70% of RGB FOV)
        self.rgb_width = 4096
        self.rgb_height = 3072
        self.thermal_width = 640
        self.thermal_height = 512
        
        # Thermal FOV is approximately 70% of RGB FOV (centered)
        self.thermal_fov_ratio = 0.7
        
        # Calculate how many RGB pixels the thermal FOV actually covers
        thermal_width_in_rgb = self.rgb_width * self.thermal_fov_ratio  # ~2867 pixels
        thermal_height_in_rgb = self.rgb_height * self.thermal_fov_ratio  # ~2150 pixels
        
        # Scale factors: how many RGB pixels per thermal pixel
        self.scale_x = thermal_width_in_rgb / self.thermal_width  # ~4.48
        self.scale_y = thermal_height_in_rgb / self.thermal_height  # ~4.20
        
        # Calculate offsets to center thermal FOV in RGB
        self.offset_x = (self.rgb_width - thermal_width_in_rgb) / 2
        self.offset_y = (self.rgb_height - thermal_height_in_rgb) / 2
        
        print(f"Alignment initialized:")
        print(f"  Thermal FOV in RGB: {thermal_width_in_rgb:.0f} x {thermal_height_in_rgb:.0f} pixels")
        print(f"  Offset: ({self.offset_x:.0f}, {self.offset_y:.0f})")
        
        # Optimized color ranges for segmentation (tuned for Rapa Nui rocky shores)
        self.ocean_hsv = {'h': (180, 250), 's': (20, 255), 'v': (30, 200)}
        self.land_hsv = {'h': (40, 150), 's': (15, 255), 'v': (10, 255)}
        self.wave_hsv = {'s': (0, 25), 'v': (150, 255)}  # Updated thresholds
        
        # Optimized thresholds for rock detection
        self.dark_threshold = 82  # Brightness below this is considered dark/rocky
        self.color_variance_threshold = 40  # Color variance for neutral detection
        self.blue_ratio_threshold = 1.20  # Blue dominance threshold
    
    def extract_aligned_rgb(self, rgb_full):
        """Extract RGB region that corresponds to thermal FOV"""
        # Calculate boundaries
        x_min = int(self.offset_x)
        y_min = int(self.offset_y)
        x_max = int(x_min + self.thermal_width * self.scale_x)
        y_max = int(y_min + self.thermal_height * self.scale_y)
        
        # Ensure within bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.rgb_width, x_max)
        y_max = min(self.rgb_height, y_max)
        
        # Extract and resize to thermal dimensions
        rgb_region = rgb_full[y_min:y_max, x_min:x_max]
        
        # Resize to match thermal
        rgb_aligned = np.array(Image.fromarray(rgb_region).resize(
            (self.thermal_width, self.thermal_height), 
            Image.Resampling.BILINEAR
        ))
        
        return rgb_aligned
    
    def load_frame_data(self, frame_number):
        """Load and align RGB and thermal data"""
        # Load RGB
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB not found: {rgb_path}")
        
        rgb_full = np.array(Image.open(rgb_path))
        rgb_aligned = self.extract_aligned_rgb(rgb_full)
        
        # Load thermal
        irg_path = self.base_path / f"IRX_{frame_number:04d}.irg"
        if irg_path.exists():
            with open(irg_path, 'rb') as f:
                irg_data = f.read()
            
            # Parse thermal data
            pixel_data_size = self.thermal_width * self.thermal_height * 2
            header_size = len(irg_data) - pixel_data_size
            
            if header_size > 0:
                raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
            else:
                raw_thermal = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
            
            raw_thermal = raw_thermal.reshape((self.thermal_height, self.thermal_width))
            temp_celsius = (raw_thermal / 10.0) - 273.15
        else:
            raise FileNotFoundError(f"Thermal not found: {irg_path}")
        
        return {
            'frame_number': frame_number,
            'rgb_full': rgb_full,
            'rgb_aligned': rgb_aligned,
            'thermal': temp_celsius,
            'raw_thermal': raw_thermal
        }
    
    def segment_ocean_land_waves(self, rgb_aligned):
        """Segment aligned RGB into ocean, land, and waves with ML or rule-based methods"""
        
        # Try ML segmentation first if available
        if self.ml_segmenter is not None:
            try:
                # ML segmenter expects uint8
                if rgb_aligned.dtype != np.uint8:
                    rgb_input = (rgb_aligned * 255).astype(np.uint8) if rgb_aligned.max() <= 1 else rgb_aligned.astype(np.uint8)
                else:
                    rgb_input = rgb_aligned
                
                # Use ultra-fast segmentation for real-time processing
                # Check if we have the fast version
                if hasattr(self.ml_segmenter, 'segment_ultra_fast'):
                    masks = self.ml_segmenter.segment_ultra_fast(rgb_input, downsample=3)
                else:
                    # Fall back to standard fast method
                    masks = self.ml_segmenter.segment_fast(rgb_input, stride=3)
                
                if masks is not None:
                    return masks
                else:
                    print("  ML segmentation failed, falling back to rule-based")
            except Exception as e:
                print(f"  ML segmentation error: {e}, falling back to rule-based")
        
        # Fall back to rule-based segmentation
        # Convert to HSV
        hsv = color.rgb2hsv(rgb_aligned)
        h = hsv[:, :, 0] * 360
        s = hsv[:, :, 1] * 255
        v = hsv[:, :, 2] * 255
        
        # Get RGB channels
        r = rgb_aligned[:,:,0].astype(float)
        g = rgb_aligned[:,:,1].astype(float)
        b = rgb_aligned[:,:,2].astype(float)
        
        # Calculate color characteristics
        max_channel = np.maximum(r, np.maximum(g, b))
        min_channel = np.minimum(r, np.minimum(g, b))
        color_range = max_channel - min_channel
        
        # Blue dominance (how much more blue than other channels)
        blue_dominance = np.zeros_like(b)
        mask = (r > 0) & (g > 0)
        blue_dominance[mask] = b[mask] / np.maximum(r[mask], g[mask])
        
        # 1. WAVE DETECTION FIRST (white foam - high brightness, low saturation)
        wave_mask = (
            (s < self.wave_hsv['s'][1]) &  # Low saturation (white/gray)
            (v > self.wave_hsv['v'][0])    # High brightness
        )
        
        # 2. ROCKY SHORE DETECTION (dark, non-blue areas)
        # Rocks are dark but NOT distinctly blue
        dark_mask = (v < self.dark_threshold)  # Dark areas
        not_blue_dominant = blue_dominance < self.blue_ratio_threshold  # Not strongly blue
        
        # Dark areas that aren't blue = rocks/shore
        rock_mask = dark_mask & not_blue_dominant
        
        # 3. OCEAN DETECTION (blue water, excluding waves and rocks)
        ocean_mask = (
            # Traditional blue water
            ((h >= 180) & (h <= 250) &  # Blue hue range
             (s >= 30) &  # Some saturation (not white)
             (v >= 40) & (v <= 200))  # Not too dark or bright
            |
            # Dark blue water (shadows, deep water)
            (dark_mask & (blue_dominance > 1.3) & (b > 30))
        )
        
        # Make sure ocean doesn't include waves or rocks
        ocean_mask = ocean_mask & ~wave_mask & ~rock_mask
        
        # 4. LAND (everything else, including vegetation and rocks)
        land_mask = ~(ocean_mask | wave_mask)
        
        # Clean up small regions
        ocean_mask = morphology.remove_small_objects(ocean_mask, min_size=100)
        land_mask = morphology.remove_small_objects(land_mask, min_size=100)
        wave_mask = morphology.remove_small_objects(wave_mask, min_size=25)
        
        # Fill small holes
        ocean_mask = morphology.remove_small_holes(ocean_mask, area_threshold=200)
        land_mask = morphology.remove_small_holes(land_mask, area_threshold=200)
        
        return {
            'ocean': ocean_mask,
            'land': land_mask,
            'waves': wave_mask
        }
    
    def detect_shoreline(self, masks):
        """Detect shoreline boundary"""
        land_dilated = morphology.binary_dilation(masks['land'], morphology.disk(2))
        shoreline = land_dilated & masks['ocean']
        
        if 'waves' in masks:
            shoreline = shoreline | masks['waves']
        
        shoreline_thin = morphology.skeletonize(shoreline)
        
        return shoreline, shoreline_thin
    
    def detect_sgd_plumes(self, thermal, masks):
        """Detect SGD plumes in thermal data"""
        # Get ocean statistics
        ocean_temps = thermal[masks['ocean']]
        
        if len(ocean_temps) == 0:
            return np.zeros_like(thermal, dtype=bool), [], {}
        
        # Robust statistics
        ocean_median = np.nanmedian(ocean_temps)
        ocean_q25 = np.nanpercentile(ocean_temps, 25)
        ocean_q75 = np.nanpercentile(ocean_temps, 75)
        ocean_iqr = ocean_q75 - ocean_q25
        
        # SGD threshold
        threshold = ocean_median - self.temp_threshold
        
        # Detect cold anomalies
        cold_mask = (thermal < threshold) & masks['ocean']
        
        # Get shoreline
        shoreline, shoreline_thin = self.detect_shoreline(masks)
        
        # Distance from shore
        distance_from_shore = ndimage.distance_transform_edt(~shoreline)
        distance_from_shore[~masks['ocean']] = np.inf
        
        # Find connected cold regions
        labeled_cold, num_features = measure.label(cold_mask, return_num=True)
        
        sgd_mask = np.zeros_like(cold_mask, dtype=bool)
        plume_info = []
        
        for i in range(1, num_features + 1):
            plume = labeled_cold == i
            
            # Check if near shore
            plume_distances = distance_from_shore[plume]
            min_shore_distance = np.nanmin(plume_distances)
            
            # Criteria for SGD
            if min_shore_distance < 5 and plume.sum() >= self.min_area:
                sgd_mask = sgd_mask | plume
                
                props = measure.regionprops(plume.astype(int))[0]
                plume_info.append({
                    'id': i,
                    'area_pixels': plume.sum(),
                    'min_shore_distance': min_shore_distance,
                    'centroid': props.centroid,
                    'bbox': props.bbox,
                    'eccentricity': props.eccentricity
                })
        
        # Calculate characteristics
        characteristics = {}
        if sgd_mask.any():
            sgd_temps = thermal[sgd_mask]
            characteristics = {
                'mean_temp': float(np.mean(sgd_temps)),
                'min_temp': float(np.min(sgd_temps)),
                'max_temp': float(np.max(sgd_temps)),
                'temp_anomaly': float(np.mean(sgd_temps) - ocean_median),
                'area_pixels': int(sgd_mask.sum()),
                'area_m2': float(sgd_mask.sum() * 0.01)  # Assuming 10cm resolution
            }
        
        return sgd_mask, plume_info, characteristics
    
    def process_frame(self, frame_number, visualize=False):
        """Process a single frame for SGD detection"""
        print(f"\nProcessing frame {frame_number}...")
        
        # Load data
        data = self.load_frame_data(frame_number)
        
        # Segment ocean/land/waves
        masks = self.segment_ocean_land_waves(data['rgb_aligned'])
        
        # Detect SGD
        sgd_mask, plume_info, characteristics = self.detect_sgd_plumes(
            data['thermal'], masks
        )
        
        print(f"  Found {len(plume_info)} SGD plumes")
        if characteristics:
            print(f"  Temperature anomaly: {characteristics['temp_anomaly']:.1f}°C")
            print(f"  Total area: {characteristics['area_m2']:.1f} m²")
        
        result = {
            'frame_number': frame_number,
            'data': data,
            'masks': masks,
            'sgd_mask': sgd_mask,
            'plume_info': plume_info,
            'characteristics': characteristics
        }
        
        if visualize:
            self.visualize_detection(result)
        
        return result
    
    def visualize_detection(self, result):
        """Visualize SGD detection results"""
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        
        # Show alignment box on full RGB
        axes[0, 0].imshow(result['data']['rgb_full'])
        rect = Rectangle(
            (self.offset_x, self.offset_y),
            self.thermal_width * self.scale_x,
            self.thermal_height * self.scale_y,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        axes[0, 0].add_patch(rect)
        axes[0, 0].set_title('Full RGB with Thermal FOV')
        axes[0, 0].axis('off')
        
        # Aligned RGB
        axes[0, 1].imshow(result['data']['rgb_aligned'])
        axes[0, 1].set_title('Aligned RGB (Thermal FOV)')
        axes[0, 1].axis('off')
        
        # Segmentation
        mask_display = np.zeros((*result['masks']['ocean'].shape, 3))
        mask_display[result['masks']['ocean']] = [0, 0.3, 1]
        mask_display[result['masks']['land']] = [0, 0.5, 0]
        mask_display[result['masks']['waves']] = [1, 1, 1]
        axes[0, 2].imshow(mask_display)
        axes[0, 2].set_title('Segmentation')
        axes[0, 2].axis('off')
        
        # Thermal
        im1 = axes[0, 3].imshow(result['data']['thermal'], cmap='RdYlBu_r')
        axes[0, 3].set_title('Thermal Data')
        axes[0, 3].axis('off')
        plt.colorbar(im1, ax=axes[0, 3], fraction=0.046)
        
        # Ocean thermal
        ocean_thermal = result['data']['thermal'].copy()
        ocean_thermal[~result['masks']['ocean']] = np.nan
        im2 = axes[1, 0].imshow(ocean_thermal, cmap='viridis')
        axes[1, 0].set_title('Ocean Thermal')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
        
        # SGD detection
        sgd_display = np.zeros((*result['sgd_mask'].shape, 3))
        sgd_display[result['sgd_mask']] = [0, 1, 1]  # Cyan for SGD
        
        # Add shoreline
        shoreline, _ = self.detect_shoreline(result['masks'])
        sgd_display[shoreline] = [1, 1, 0]  # Yellow for shore
        
        axes[1, 1].imshow(sgd_display)
        axes[1, 1].set_title(f'SGD Plumes ({len(result["plume_info"])} detected)')
        axes[1, 1].axis('off')
        
        # Overlay on thermal
        thermal_norm = (result['data']['thermal'] - np.nanmin(result['data']['thermal'])) / \
                      (np.nanmax(result['data']['thermal']) - np.nanmin(result['data']['thermal']))
        overlay = plt.cm.RdYlBu_r(thermal_norm)[:,:,:3]
        overlay[result['sgd_mask']] = [0, 1, 0]  # Green for SGD
        
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('Thermal + SGD Overlay')
        axes[1, 2].axis('off')
        
        # Statistics
        stats_text = f"Frame {result['frame_number']} Statistics:\n\n"
        
        # Coverage
        total_px = result['masks']['ocean'].size
        stats_text += f"Ocean: {100*result['masks']['ocean'].sum()/total_px:.1f}%\n"
        stats_text += f"Land: {100*result['masks']['land'].sum()/total_px:.1f}%\n"
        stats_text += f"Waves: {100*result['masks']['waves'].sum()/total_px:.1f}%\n\n"
        
        # Temperature
        ocean_temps = result['data']['thermal'][result['masks']['ocean']]
        if len(ocean_temps) > 0:
            stats_text += f"Ocean Temperature:\n"
            stats_text += f"Mean: {np.mean(ocean_temps):.1f}°C\n"
            stats_text += f"Std: {np.std(ocean_temps):.2f}°C\n\n"
        
        # SGD
        stats_text += f"SGD Detection:\n"
        stats_text += f"Plumes found: {len(result['plume_info'])}\n"
        if result['characteristics']:
            stats_text += f"Temp anomaly: {result['characteristics']['temp_anomaly']:.1f}°C\n"
            stats_text += f"Total area: {result['characteristics']['area_m2']:.1f} m²"
        
        axes[1, 3].text(0.05, 0.5, stats_text, transform=axes[1, 3].transAxes,
                       fontsize=10, verticalalignment='center')
        axes[1, 3].axis('off')
        
        plt.suptitle(f'SGD Detection - Frame {result["frame_number"]:04d}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def batch_process(self, frame_numbers=None, output_dir="sgd_output", max_frames=10, 
                     georeference=True):
        """Process multiple frames with optional georeferencing"""
        if frame_numbers is None:
            # Find available frames
            rgb_files = sorted(self.base_path.glob("MAX_*.JPG"))[:max_frames]
            frame_numbers = []
            
            for rgb_file in rgb_files:
                num = int(rgb_file.stem.split('_')[1])
                if (self.base_path / f"IRX_{num:04d}.irg").exists():
                    frame_numbers.append(num)
        
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Initialize georeferencer if requested
        georef = None
        if georeference:
            try:
                from sgd_georef_simple import SimpleGeoref
                georef = SimpleGeoref(self.base_path)
                print("✅ Georeferencing enabled")
            except ImportError:
                print("⚠️ Georeferencing module not found, continuing without GPS")
                georeference = False
        
        for frame_num in frame_numbers:
            try:
                result = self.process_frame(frame_num, visualize=False)
                results.append(result)
                
                # Save visualization
                fig = self.visualize_detection(result)
                fig.savefig(output_path / f"sgd_frame_{frame_num:04d}.png", dpi=150)
                plt.close()
                
                # Add georeferencing if available
                if georeference and georef and len(result['plume_info']) > 0:
                    georef.process_sgd_frame(frame_num, result['plume_info'])
                
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
        
        # Export georeferenced data if available
        if georeference and georef and len(georef.sgd_locations) > 0:
            print(f"\n📍 Exporting {len(georef.sgd_locations)} georeferenced SGD locations...")
            georef.export_to_geojson(str(output_path / "sgd_locations.geojson"))
            georef.export_to_csv(str(output_path / "sgd_locations.csv"))
            georef.export_to_kml(str(output_path / "sgd_locations.kml"))
            print("  ✅ Created sgd_locations.geojson (for GIS)")
            print("  ✅ Created sgd_locations.csv (spreadsheet)")
            print("  ✅ Created sgd_locations.kml (for Google Earth)")
        
        # Create summary
        self.create_summary(results, output_path)
        
        return results
    
    def create_summary(self, results, output_path):
        """Create summary of all detections"""
        if not results:
            return
        
        # Aggregate statistics
        total_plumes = sum(len(r['plume_info']) for r in results)
        frames_with_sgd = sum(1 for r in results if len(r['plume_info']) > 0)
        
        print(f"\n{'='*50}")
        print(f"SGD Detection Summary")
        print(f"{'='*50}")
        print(f"Frames processed: {len(results)}")
        print(f"Frames with SGD: {frames_with_sgd}")
        print(f"Total plumes detected: {total_plumes}")
        
        if total_plumes > 0:
            avg_anomaly = np.mean([r['characteristics']['temp_anomaly'] 
                                  for r in results if r['characteristics']])
            total_area = sum(r['characteristics'].get('area_m2', 0) 
                           for r in results if r['characteristics'])
            print(f"Average temperature anomaly: {avg_anomaly:.1f}°C")
            print(f"Total SGD area: {total_area:.1f} m²")
        
        # Save summary
        summary = {
            'frames_processed': len(results),
            'frames_with_sgd': frames_with_sgd,
            'total_plumes': total_plumes,
            'frame_details': []
        }
        
        for r in results:
            summary['frame_details'].append({
                'frame': r['frame_number'],
                'num_plumes': len(r['plume_info']),
                'characteristics': r['characteristics']
            })
        
        with open(output_path / 'sgd_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {output_path}/")


def main():
    """Main entry point with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Integrated SGD Detector with ML Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive menu (default)
  python sgd_detector_integrated.py
  
  # Use custom ML model
  python sgd_detector_integrated.py --model cloudy_conditions.pkl
  
  # Disable ML, use rule-based segmentation
  python sgd_detector_integrated.py --no-ml
  
  # Direct mode selection
  python sgd_detector_integrated.py --mode single --frame 250
  python sgd_detector_integrated.py --mode batch --start 200 --end 300
  python sgd_detector_integrated.py --mode interactive
        """
    )
    
    parser.add_argument('--model', type=str, default='segmentation_model.pkl',
                       help='Path to ML segmentation model (default: segmentation_model.pkl)')
    parser.add_argument('--no-ml', action='store_true',
                       help='Disable ML segmentation, use rule-based instead')
    parser.add_argument('--mode', choices=['single', 'batch', 'interactive'],
                       help='Operating mode (if not specified, shows menu)')
    parser.add_argument('--frame', type=int, default=248,
                       help='Frame number for single mode (default: 248)')
    parser.add_argument('--start', type=int, default=1,
                       help='Start frame for batch mode (default: 1)')
    parser.add_argument('--end', type=int, default=100,
                       help='End frame for batch mode (default: 100)')
    
    args = parser.parse_args()
    
    print("Integrated SGD Detector")
    print("=" * 50)
    print("\nBuilt-in alignment for Autel 640T:")
    print("  RGB: 4096 x 3072 pixels")
    print("  Thermal: 640 x 512 pixels")
    print("  Thermal FOV is centered in RGB image")
    
    # Initialize detector with specified model
    ml_model_path = None if args.no_ml else args.model
    detector = IntegratedSGDDetector(
        use_ml=ml_model_path is not None,
        ml_model_path=ml_model_path if ml_model_path else "segmentation_model.pkl"
    )
    
    # Direct mode execution if specified
    if args.mode:
        choice = {'single': '1', 'batch': '2', 'interactive': '3'}[args.mode]
    else:
        print("\nOptions:")
        print("1. Single frame analysis")
        print("2. Batch process frames")
        print("3. Interactive parameter tuning")
        choice = input("\nChoice (1-3): ").strip()
    
    if choice == '1':
        # Use command-line frame if in direct mode, otherwise prompt
        if args.mode == 'single':
            frame = args.frame
        else:
            frame = int(input("Enter frame number (default 248): ") or "248")
        result = detector.process_frame(frame, visualize=True)
        
    elif choice == '2':
        # Use command-line range if in direct mode, otherwise prompt
        if args.mode == 'batch':
            max_frames = args.end - args.start + 1
        else:
            max_frames = int(input("Max frames to process (default 10): ") or "10")
        results = detector.batch_process(max_frames=max_frames)
        
    else:
        # Interactive viewer with proper navigation
        print("\nInteractive mode...")
        
        # Find available frames
        frames = []
        for f in sorted(detector.base_path.glob("MAX_*.JPG"))[:50]:  # Check more frames
            num = int(f.stem.split('_')[1])
            if (detector.base_path / f"IRX_{num:04d}.irg").exists():
                frames.append(num)
        
        if not frames:
            print("No data found!")
            return
        
        print(f"Found {len(frames)} frames: {frames[0]} to {frames[-1]}")
        
        from matplotlib.widgets import Slider, Button
        
        # Create figure with specific subplot layout
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid for main visualizations (leave bottom space for controls)
        gs = fig.add_gridspec(2, 4, bottom=0.20, top=0.92, hspace=0.3, wspace=0.3)
        axes = []
        for i in range(2):
            for j in range(4):
                axes.append(fig.add_subplot(gs[i, j]))
        
        # State variables
        current = {'frame_idx': 0, 'temp_threshold': 1.0, 'min_area': 50}
        
        def update_display():
            """Update the display with current frame and parameters"""
            # Clear all axes
            for ax in axes:
                ax.clear()
            
            # Update detector parameters
            detector.temp_threshold = current['temp_threshold']
            detector.min_area = current['min_area']
            
            # Get current frame
            frame_num = frames[current['frame_idx']]
            
            # Process frame
            try:
                result = detector.process_frame(frame_num, visualize=False)
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
                return
            
            # Update title
            fig.suptitle(f'SGD Detection - Frame {frame_num:04d} ({current["frame_idx"]+1}/{len(frames)})', 
                        fontsize=14)
            
            # Recreate visualization in the axes
            # Row 1
            axes[0].imshow(result['data']['rgb_full'])
            rect = Rectangle(
                (detector.offset_x, detector.offset_y),
                detector.thermal_width * detector.scale_x,
                detector.thermal_height * detector.scale_y,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].set_title('Full RGB with Thermal FOV')
            axes[0].axis('off')
            
            axes[1].imshow(result['data']['rgb_aligned'])
            axes[1].set_title('Aligned RGB')
            axes[1].axis('off')
            
            mask_display = np.zeros((*result['masks']['ocean'].shape, 3))
            mask_display[result['masks']['ocean']] = [0, 0.3, 1]
            mask_display[result['masks']['land']] = [0, 0.5, 0]
            mask_display[result['masks']['waves']] = [1, 1, 1]
            axes[2].imshow(mask_display)
            axes[2].set_title('Segmentation')
            axes[2].axis('off')
            
            im1 = axes[3].imshow(result['data']['thermal'], cmap='RdYlBu_r')
            axes[3].set_title('Thermal')
            axes[3].axis('off')
            
            # Row 2
            ocean_thermal = result['data']['thermal'].copy()
            ocean_thermal[~result['masks']['ocean']] = np.nan
            im2 = axes[4].imshow(ocean_thermal, cmap='viridis')
            axes[4].set_title('Ocean Thermal')
            axes[4].axis('off')
            
            sgd_display = np.zeros((*result['sgd_mask'].shape, 3))
            sgd_display[result['sgd_mask']] = [0, 1, 1]
            shoreline, _ = detector.detect_shoreline(result['masks'])
            sgd_display[shoreline] = [1, 1, 0]
            axes[5].imshow(sgd_display)
            axes[5].set_title(f'SGD: {len(result["plume_info"])} plumes')
            axes[5].axis('off')
            
            thermal_norm = (result['data']['thermal'] - np.nanmin(result['data']['thermal'])) / \
                          (np.nanmax(result['data']['thermal']) - np.nanmin(result['data']['thermal']) + 1e-10)
            overlay = plt.cm.RdYlBu_r(thermal_norm)[:,:,:3]
            overlay[result['sgd_mask']] = [0, 1, 0]
            axes[6].imshow(overlay)
            axes[6].set_title('Thermal + SGD')
            axes[6].axis('off')
            
            # Statistics
            stats_text = f"Statistics:\n\n"
            total_px = result['masks']['ocean'].size
            stats_text += f"Ocean: {100*result['masks']['ocean'].sum()/total_px:.1f}%\n"
            stats_text += f"Land: {100*result['masks']['land'].sum()/total_px:.1f}%\n"
            stats_text += f"Waves: {100*result['masks']['waves'].sum()/total_px:.1f}%\n\n"
            
            stats_text += f"Parameters:\n"
            stats_text += f"Temp threshold: {current['temp_threshold']:.1f}°C\n"
            stats_text += f"Min area: {current['min_area']} px\n\n"
            
            stats_text += f"SGD Detection:\n"
            stats_text += f"Plumes: {len(result['plume_info'])}\n"
            if result['characteristics']:
                stats_text += f"Anomaly: {result['characteristics']['temp_anomaly']:.1f}°C\n"
                stats_text += f"Area: {result['characteristics']['area_m2']:.1f} m²"
            
            axes[7].text(0.05, 0.5, stats_text, transform=axes[7].transAxes,
                        fontsize=10, verticalalignment='center')
            axes[7].axis('off')
            
            fig.canvas.draw_idle()
        
        # Navigation buttons
        ax_prev = plt.axes([0.10, 0.12, 0.08, 0.04])
        btn_prev = Button(ax_prev, '← Previous')
        
        ax_next = plt.axes([0.19, 0.12, 0.08, 0.04])
        btn_next = Button(ax_next, 'Next →')
        
        ax_first = plt.axes([0.28, 0.12, 0.08, 0.04])
        btn_first = Button(ax_first, 'First')
        
        ax_last = plt.axes([0.37, 0.12, 0.08, 0.04])
        btn_last = Button(ax_last, 'Last')
        
        # Save button
        ax_save = plt.axes([0.85, 0.12, 0.08, 0.04])
        btn_save = Button(ax_save, 'Save Fig')
        
        # Frame slider (for quick navigation)
        ax_frame = plt.axes([0.10, 0.07, 0.35, 0.03])
        slider_frame = Slider(ax_frame, 'Frame', 0, len(frames)-1, 
                            valinit=0, valstep=1)
        
        # Parameter sliders
        ax_temp = plt.axes([0.55, 0.12, 0.25, 0.03])
        slider_temp = Slider(ax_temp, 'Temp Δ (°C)', 0.5, 3.0, 
                           valinit=1.0, valstep=0.1)
        
        ax_area = plt.axes([0.55, 0.07, 0.25, 0.03])
        slider_area = Slider(ax_area, 'Min Area (px)', 10, 200, 
                           valinit=50, valstep=10)
        
        # Frame info text
        ax_info = plt.axes([0.47, 0.12, 0.06, 0.04])
        ax_info.axis('off')
        frame_info = ax_info.text(0.5, 0.5, f'Frame {current["frame_idx"]+1}/{len(frames)}', 
                                  ha='center', va='center', fontsize=12)
        
        # Button callbacks
        def next_frame(event):
            if current['frame_idx'] < len(frames) - 1:
                current['frame_idx'] += 1
                slider_frame.set_val(current['frame_idx'])
                frame_info.set_text(f'Frame {current["frame_idx"]+1}/{len(frames)}')
                update_display()
        
        def prev_frame(event):
            if current['frame_idx'] > 0:
                current['frame_idx'] -= 1
                slider_frame.set_val(current['frame_idx'])
                frame_info.set_text(f'Frame {current["frame_idx"]+1}/{len(frames)}')
                update_display()
        
        def first_frame(event):
            current['frame_idx'] = 0
            slider_frame.set_val(current['frame_idx'])
            frame_info.set_text(f'Frame {current["frame_idx"]+1}/{len(frames)}')
            update_display()
        
        def last_frame(event):
            current['frame_idx'] = len(frames) - 1
            slider_frame.set_val(current['frame_idx'])
            frame_info.set_text(f'Frame {current["frame_idx"]+1}/{len(frames)}')
            update_display()
        
        def save_figure(event):
            frame_num = frames[current['frame_idx']]
            filename = f'sgd_frame_{frame_num:04d}_interactive.png'
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved to {filename}")
        
        # Slider callbacks
        def update_frame_slider(val):
            current['frame_idx'] = int(slider_frame.val)
            frame_info.set_text(f'Frame {current["frame_idx"]+1}/{len(frames)}')
            update_display()
        
        def update_temp(val):
            current['temp_threshold'] = slider_temp.val
            update_display()
        
        def update_area(val):
            current['min_area'] = int(slider_area.val)
            update_display()
        
        # Connect callbacks
        btn_next.on_clicked(next_frame)
        btn_prev.on_clicked(prev_frame)
        btn_first.on_clicked(first_frame)
        btn_last.on_clicked(last_frame)
        btn_save.on_clicked(save_figure)
        
        slider_frame.on_changed(update_frame_slider)
        slider_temp.on_changed(update_temp)
        slider_area.on_changed(update_area)
        
        # Keyboard shortcuts
        def on_key(event):
            if event.key == 'right':
                next_frame(None)
            elif event.key == 'left':
                prev_frame(None)
            elif event.key == 'home':
                first_frame(None)
            elif event.key == 'end':
                last_frame(None)
            elif event.key == 's':
                save_figure(None)
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        print("\nControls:")
        print("  Buttons: Previous, Next, First, Last, Save")
        print("  Keyboard: ← → (navigate), Home/End (first/last), S (save)")
        print("  Sliders: Adjust temperature threshold and minimum area")
        
        # Initial display
        update_display()
        
        plt.show()


if __name__ == "__main__":
    main()