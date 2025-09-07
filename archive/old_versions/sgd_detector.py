#!/usr/bin/env python3
"""
Submarine Groundwater Discharge (SGD) Detector
Identifies and maps cold freshwater seeps along the shoreline using thermal imagery
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from PIL import Image
from pathlib import Path
from scipy import ndimage, signal, spatial
from skimage import filters, morphology, measure, feature, color
import json
from datetime import datetime
from rgb_ocean_segmenter import RGBOceanSegmenter
import warnings
warnings.filterwarnings('ignore')


class SGDDetector:
    """Detect submarine groundwater discharge from thermal imagery"""
    
    def __init__(self, temp_threshold=1.0, min_area=50):
        """
        Initialize SGD detector
        
        Parameters:
        - temp_threshold: Minimum temperature difference (°C) below ocean mean to consider as SGD
        - min_area: Minimum area (pixels) for a valid SGD plume
        """
        self.temp_threshold = temp_threshold
        self.min_area = min_area
        self.rgb_segmenter = RGBOceanSegmenter()
        
    def detect_shoreline(self, masks):
        """
        Detect shoreline as boundary between ocean and land
        """
        # Get ocean-land boundary
        ocean_mask = masks['ocean']
        land_mask = masks['land']
        
        # Dilate land slightly to ensure we capture the interface
        land_dilated = morphology.binary_dilation(land_mask, morphology.disk(2))
        
        # Shoreline is where dilated land meets ocean
        shoreline = land_dilated & ocean_mask
        
        # Also include wave zone as potential SGD area
        if 'waves' in masks:
            shoreline = shoreline | masks['waves']
        
        # Thin the shoreline to get a line
        shoreline_thin = morphology.skeletonize(shoreline)
        
        return shoreline, shoreline_thin
    
    def create_shore_distance_map(self, ocean_mask, shoreline):
        """
        Create distance map from shoreline for each ocean pixel
        """
        # Distance transform from shoreline
        distance_from_shore = ndimage.distance_transform_edt(~shoreline)
        
        # Mask to ocean only
        distance_from_shore[~ocean_mask] = np.nan
        
        return distance_from_shore
    
    def detect_cold_anomalies(self, thermal, ocean_mask, adaptive=True):
        """
        Detect cold anomalies in ocean that could be SGD
        """
        # Get ocean temperatures
        ocean_temps = thermal[ocean_mask]
        
        if len(ocean_temps) == 0:
            return np.zeros_like(thermal, dtype=bool)
        
        # Calculate statistics
        ocean_mean = np.nanmean(ocean_temps)
        ocean_std = np.nanstd(ocean_temps)
        ocean_median = np.nanmedian(ocean_temps)
        
        # Use robust statistics to avoid outlier influence
        ocean_q25 = np.nanpercentile(ocean_temps, 25)
        ocean_q75 = np.nanpercentile(ocean_temps, 75)
        ocean_iqr = ocean_q75 - ocean_q25
        
        print(f"Ocean stats: mean={ocean_mean:.2f}°C, median={ocean_median:.2f}°C, IQR={ocean_iqr:.2f}°C")
        
        if adaptive:
            # Adaptive threshold based on local statistics
            # SGD is typically 2-5°C cooler than ambient ocean
            threshold_temp = ocean_median - self.temp_threshold
            
            # Also consider statistical outliers
            outlier_threshold = ocean_q25 - 1.5 * ocean_iqr  # Below lower whisker
            
            # Use the more conservative threshold
            threshold = max(threshold_temp, outlier_threshold)
        else:
            # Fixed threshold
            threshold = ocean_mean - self.temp_threshold
        
        print(f"Cold threshold: {threshold:.2f}°C")
        
        # Detect cold pixels
        cold_mask = (thermal < threshold) & ocean_mask
        
        return cold_mask, threshold, ocean_median
    
    def track_plumes_from_shore(self, cold_mask, shoreline, distance_map, max_distance=100):
        """
        Track cold water plumes emanating from shoreline
        """
        # Label connected components in cold mask
        labeled_cold, num_features = measure.label(cold_mask, return_num=True)
        
        valid_plumes = np.zeros_like(cold_mask, dtype=bool)
        plume_info = []
        
        for i in range(1, num_features + 1):
            plume = labeled_cold == i
            
            # Check if plume is connected to or near shoreline
            plume_distances = distance_map[plume]
            min_shore_distance = np.nanmin(plume_distances)
            
            # Criteria for valid SGD plume:
            # 1. Must be near shore (within 5 pixels)
            # 2. Must have minimum area
            # 3. Should extend from shore (not isolated spot)
            
            if min_shore_distance < 5 and plume.sum() >= self.min_area:
                # Additional shape analysis
                props = measure.regionprops(plume.astype(int))[0]
                
                # Eccentricity: 0 = circle, 1 = line
                # SGD plumes tend to be elongated
                eccentricity = props.eccentricity
                
                # Orientation relative to shore
                orientation = props.orientation
                
                # Distance distribution - should decrease from shore
                mean_distance = np.nanmean(plume_distances)
                
                # Accept if it looks like a plume
                if mean_distance < max_distance:
                    valid_plumes = valid_plumes | plume
                    
                    plume_info.append({
                        'id': i,
                        'area_pixels': plume.sum(),
                        'min_shore_distance': min_shore_distance,
                        'mean_shore_distance': mean_distance,
                        'eccentricity': eccentricity,
                        'orientation': np.degrees(orientation),
                        'centroid': props.centroid,
                        'bbox': props.bbox
                    })
        
        return valid_plumes, plume_info
    
    def analyze_plume_characteristics(self, thermal, plume_mask, ocean_median):
        """
        Analyze thermal characteristics of detected plumes
        """
        if not plume_mask.any():
            return {}
        
        plume_temps = thermal[plume_mask]
        
        characteristics = {
            'mean_temp': float(np.nanmean(plume_temps)),
            'min_temp': float(np.nanmin(plume_temps)),
            'max_temp': float(np.nanmax(plume_temps)),
            'std_temp': float(np.nanstd(plume_temps)),
            'temp_anomaly': float(np.nanmean(plume_temps) - ocean_median),
            'area_m2': float(plume_mask.sum() * 0.01),  # Assuming ~10cm pixel resolution
            'num_pixels': int(plume_mask.sum())
        }
        
        return characteristics
    
    def create_sgd_confidence_map(self, thermal, cold_mask, shoreline, distance_map):
        """
        Create confidence map for SGD detection
        Higher confidence near shore with consistent cold signature
        """
        confidence = np.zeros_like(thermal)
        
        if not cold_mask.any():
            return confidence
        
        # Base confidence from temperature anomaly
        ocean_median = np.nanmedian(thermal[~np.isnan(thermal)])
        temp_anomaly = ocean_median - thermal
        temp_anomaly[temp_anomaly < 0] = 0
        
        # Normalize to 0-1
        if temp_anomaly.max() > 0:
            temp_confidence = temp_anomaly / temp_anomaly.max()
        else:
            temp_confidence = temp_anomaly
        
        # Distance weighting - higher confidence near shore
        distance_weight = np.exp(-distance_map / 20)  # Exponential decay
        distance_weight[np.isnan(distance_weight)] = 0
        
        # Combine factors
        confidence = temp_confidence * distance_weight * cold_mask
        
        # Smooth to reduce noise
        confidence = ndimage.gaussian_filter(confidence, sigma=2)
        
        return confidence
    
    def process_frame(self, frame_number, base_path="data/100MEDIA"):
        """
        Process a single frame to detect SGD
        """
        base_path = Path(base_path)
        
        print(f"\nProcessing frame {frame_number}...")
        
        # Load RGB and thermal data
        data = self.rgb_segmenter.load_rgb_thermal_pair(frame_number, base_path)
        
        # Segment ocean/land using RGB
        masks = self.rgb_segmenter.segment_rgb_image(data['rgb_resized'], method='combined')
        
        # Detect shoreline
        shoreline, shoreline_thin = self.detect_shoreline(masks)
        
        # Create distance map
        distance_map = self.create_shore_distance_map(masks['ocean'], shoreline)
        
        # Detect cold anomalies
        cold_mask, threshold, ocean_median = self.detect_cold_anomalies(
            data['thermal'], masks['ocean'], adaptive=True
        )
        
        # Track plumes from shore
        sgd_plumes, plume_info = self.track_plumes_from_shore(
            cold_mask, shoreline, distance_map
        )
        
        # Analyze characteristics
        characteristics = self.analyze_plume_characteristics(
            data['thermal'], sgd_plumes, ocean_median
        )
        
        # Create confidence map
        confidence_map = self.create_sgd_confidence_map(
            data['thermal'], cold_mask, shoreline, distance_map
        )
        
        return {
            'frame_number': frame_number,
            'thermal': data['thermal'],
            'rgb': data['rgb_resized'],
            'masks': masks,
            'shoreline': shoreline,
            'shoreline_thin': shoreline_thin,
            'distance_map': distance_map,
            'cold_mask': cold_mask,
            'sgd_plumes': sgd_plumes,
            'plume_info': plume_info,
            'characteristics': characteristics,
            'confidence_map': confidence_map,
            'threshold': threshold,
            'ocean_median': ocean_median
        }
    
    def visualize_sgd_detection(self, result):
        """
        Create comprehensive visualization of SGD detection
        """
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle(f'SGD Detection - Frame {result["frame_number"]:04d}', fontsize=14)
        
        # 1. RGB image
        axes[0, 0].imshow(result['rgb'])
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')
        
        # 2. Thermal with shoreline
        im1 = axes[0, 1].imshow(result['thermal'], cmap='RdYlBu_r')
        axes[0, 1].contour(result['shoreline_thin'], colors='white', linewidths=2)
        axes[0, 1].set_title('Thermal + Shoreline')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # 3. Distance from shore
        distance_display = result['distance_map'].copy()
        distance_display[distance_display > 100] = 100  # Cap for display
        im2 = axes[0, 2].imshow(distance_display, cmap='viridis')
        axes[0, 2].set_title('Distance from Shore')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # 4. Cold anomalies
        cold_display = np.zeros((*result['cold_mask'].shape, 3))
        cold_display[result['cold_mask']] = [0, 0, 1]  # Blue for cold
        cold_display[result['shoreline']] = [1, 1, 0]  # Yellow for shoreline
        axes[0, 3].imshow(cold_display)
        axes[0, 3].set_title(f'Cold Anomalies\n(< {result["threshold"]:.1f}°C)')
        axes[0, 3].axis('off')
        
        # 5. Detected SGD plumes
        sgd_display = np.zeros((*result['sgd_plumes'].shape, 3))
        sgd_display[result['sgd_plumes']] = [0, 1, 1]  # Cyan for SGD
        sgd_display[result['shoreline']] = [1, 1, 0]  # Yellow for shoreline
        axes[1, 0].imshow(sgd_display)
        axes[1, 0].set_title(f'SGD Plumes\n({len(result["plume_info"])} detected)')
        axes[1, 0].axis('off')
        
        # 6. Confidence map
        im3 = axes[1, 1].imshow(result['confidence_map'], cmap='hot')
        axes[1, 1].set_title('SGD Confidence')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
        
        # 7. Thermal with SGD overlay
        thermal_overlay = result['thermal'].copy()
        thermal_min = np.nanmin(thermal_overlay[result['masks']['ocean']])
        thermal_max = np.nanmax(thermal_overlay[result['masks']['ocean']])
        thermal_norm = (thermal_overlay - thermal_min) / (thermal_max - thermal_min)
        
        # Create RGB overlay
        overlay_rgb = plt.cm.RdYlBu_r(thermal_norm)[:,:,:3]
        overlay_rgb[result['sgd_plumes']] = [0, 1, 0]  # Green for SGD
        
        axes[1, 2].imshow(overlay_rgb)
        axes[1, 2].contour(result['shoreline_thin'], colors='white', linewidths=1)
        axes[1, 2].set_title('Thermal + SGD Overlay')
        axes[1, 2].axis('off')
        
        # 8. Statistics
        stats_text = "Detection Statistics:\n\n"
        stats_text += f"Ocean median temp: {result['ocean_median']:.1f}°C\n"
        stats_text += f"Cold threshold: {result['threshold']:.1f}°C\n"
        stats_text += f"SGD plumes found: {len(result['plume_info'])}\n"
        
        if result['characteristics']:
            stats_text += f"\nSGD Characteristics:\n"
            stats_text += f"Mean temp: {result['characteristics']['mean_temp']:.1f}°C\n"
            stats_text += f"Min temp: {result['characteristics']['min_temp']:.1f}°C\n"
            stats_text += f"Anomaly: {result['characteristics']['temp_anomaly']:.1f}°C\n"
            stats_text += f"Total area: {result['characteristics']['area_m2']:.1f} m²\n"
        
        if result['plume_info']:
            stats_text += f"\nLargest plume:\n"
            largest = max(result['plume_info'], key=lambda x: x['area_pixels'])
            stats_text += f"  Area: {largest['area_pixels']} pixels\n"
            stats_text += f"  Shore dist: {largest['min_shore_distance']:.1f} px\n"
        
        axes[1, 3].text(0.05, 0.5, stats_text, transform=axes[1, 3].transAxes,
                       fontsize=9, verticalalignment='center')
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        return fig
    
    def export_to_geojson(self, results, output_path="sgd_detections.geojson"):
        """
        Export SGD detections to GeoJSON for GIS import
        Note: This requires georeferencing information (GPS/IMU data)
        """
        features = []
        
        for result in results:
            if not result['plume_info']:
                continue
            
            # For each plume in frame
            for plume in result['plume_info']:
                # Convert pixel coordinates to geographic
                # This is simplified - real implementation needs GPS/IMU data
                feature = {
                    "type": "Feature",
                    "properties": {
                        "frame": result['frame_number'],
                        "area_pixels": plume['area_pixels'],
                        "area_m2": plume['area_pixels'] * 0.01,  # Assuming 10cm resolution
                        "shore_distance": plume['min_shore_distance'],
                        "confidence": float(result['confidence_map'][
                            int(plume['centroid'][0]), 
                            int(plume['centroid'][1])
                        ]),
                        "temp_anomaly": result['characteristics'].get('temp_anomaly', 0),
                        "timestamp": datetime.now().isoformat()
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[  # Placeholder - needs real coordinates
                            [plume['bbox'][1], plume['bbox'][0]],
                            [plume['bbox'][3], plume['bbox'][0]],
                            [plume['bbox'][3], plume['bbox'][2]],
                            [plume['bbox'][1], plume['bbox'][2]],
                            [plume['bbox'][1], plume['bbox'][0]]
                        ]]
                    }
                }
                features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(features)} SGD features to {output_path}")
        
        return geojson
    
    def create_sgd_composite_map(self, results):
        """
        Create composite map showing all SGD detections across frames
        """
        if not results:
            return None
        
        # Use first frame as reference
        shape = results[0]['thermal'].shape
        
        # Accumulate detections
        sgd_frequency = np.zeros(shape)
        confidence_sum = np.zeros(shape)
        
        for result in results:
            if result['sgd_plumes'].any():
                sgd_frequency[result['sgd_plumes']] += 1
                confidence_sum += result['confidence_map']
        
        # Average confidence
        confidence_avg = confidence_sum / len(results)
        
        # Persistence map - how often SGD detected at each location
        persistence = sgd_frequency / len(results)
        
        return {
            'frequency': sgd_frequency,
            'persistence': persistence,
            'confidence_avg': confidence_avg
        }


class SGDAnalyzer:
    """Interactive analyzer for SGD detection"""
    
    def __init__(self, base_path="data/100MEDIA"):
        self.detector = SGDDetector()
        self.base_path = Path(base_path)
        
    def batch_process(self, frame_numbers=None, export_gis=True):
        """
        Process multiple frames to detect SGD
        """
        if frame_numbers is None:
            # Get all available frames
            rgb_files = list(self.base_path.glob("MAX_*.JPG"))
            frame_numbers = []
            for f in rgb_files:
                num = int(f.stem.split('_')[1])
                if (self.base_path / f"IRX_{num:04d}.irg").exists() or \
                   (self.base_path / f"IRX_{num:04d}.TIFF").exists():
                    frame_numbers.append(num)
            frame_numbers = sorted(frame_numbers)[:10]  # Limit for testing
        
        results = []
        
        for frame_num in frame_numbers:
            try:
                result = self.detector.process_frame(frame_num, self.base_path)
                results.append(result)
                
                # Save individual visualization
                fig = self.detector.visualize_sgd_detection(result)
                output_dir = Path("sgd_output")
                output_dir.mkdir(exist_ok=True)
                fig.savefig(output_dir / f"sgd_frame_{frame_num:04d}.png", dpi=150)
                plt.close()
                
                print(f"Frame {frame_num}: {len(result['plume_info'])} SGD plumes detected")
                
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
        
        # Create composite map
        if results:
            composite = self.detector.create_sgd_composite_map(results)
            
            # Visualize composite
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im1 = axes[0].imshow(composite['frequency'], cmap='hot')
            axes[0].set_title('SGD Detection Frequency')
            axes[0].axis('off')
            plt.colorbar(im1, ax=axes[0])
            
            im2 = axes[1].imshow(composite['persistence'], cmap='hot', vmin=0, vmax=1)
            axes[1].set_title('SGD Persistence (0-1)')
            axes[1].axis('off')
            plt.colorbar(im2, ax=axes[1])
            
            im3 = axes[2].imshow(composite['confidence_avg'], cmap='hot')
            axes[2].set_title('Average Confidence')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2])
            
            plt.suptitle(f'SGD Composite Map - {len(results)} frames')
            plt.tight_layout()
            plt.savefig(Path("sgd_output") / "sgd_composite.png", dpi=150)
            plt.show()
            
            # Export to GeoJSON
            if export_gis:
                self.detector.export_to_geojson(results, "sgd_output/sgd_detections.geojson")
        
        return results
    
    def interactive_viewer(self):
        """
        Interactive viewer for SGD detection with adjustable parameters
        """
        # Get available frames
        rgb_files = list(self.base_path.glob("MAX_*.JPG"))
        frame_numbers = []
        for f in rgb_files:
            num = int(f.stem.split('_')[1])
            if (self.base_path / f"IRX_{num:04d}.irg").exists():
                frame_numbers.append(num)
        
        if not frame_numbers:
            print("No data found!")
            return
        
        frame_numbers = sorted(frame_numbers)
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # State
        current = {
            'frame_idx': 0,
            'temp_threshold': 1.0,
            'min_area': 50
        }
        
        # Process initial frame
        self.detector.temp_threshold = current['temp_threshold']
        self.detector.min_area = current['min_area']
        result = self.detector.process_frame(frame_numbers[0], self.base_path)
        
        # Create layout  
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        axes = []
        for i in range(3):
            for j in range(4):
                axes.append(fig.add_subplot(gs[i, j]))
        
        def update_display(val=None):
            """Update display"""
            
            # Clear axes
            for ax in axes:
                ax.clear()
            
            # Update detector parameters
            self.detector.temp_threshold = current['temp_threshold']
            self.detector.min_area = current['min_area']
            
            # Process frame
            frame_num = frame_numbers[current['frame_idx']]
            result = self.detector.process_frame(frame_num, self.base_path)
            
            fig.suptitle(f'SGD Detection - Frame {frame_num:04d}', fontsize=14)
            
            # Display results (similar to visualize_sgd_detection)
            axes[0].imshow(result['rgb'])
            axes[0].set_title('RGB')
            axes[0].axis('off')
            
            im1 = axes[1].imshow(result['thermal'], cmap='RdYlBu_r')
            axes[1].contour(result['shoreline_thin'], colors='white', linewidths=2)
            axes[1].set_title('Thermal + Shore')
            axes[1].axis('off')
            
            cold_display = np.zeros((*result['cold_mask'].shape, 3))
            cold_display[result['cold_mask']] = [0, 0, 1]
            axes[2].imshow(cold_display)
            axes[2].set_title(f'Cold < {result["threshold"]:.1f}°C')
            axes[2].axis('off')
            
            sgd_display = np.zeros((*result['sgd_plumes'].shape, 3))
            sgd_display[result['sgd_plumes']] = [0, 1, 1]
            sgd_display[result['shoreline']] = [1, 1, 0]
            axes[3].imshow(sgd_display)
            axes[3].set_title(f'{len(result["plume_info"])} SGD Plumes')
            axes[3].axis('off')
            
            # More visualizations...
            im2 = axes[4].imshow(result['confidence_map'], cmap='hot')
            axes[4].set_title('Confidence Map')
            axes[4].axis('off')
            
            # Distance map
            dist_display = result['distance_map'].copy()
            dist_display[dist_display > 50] = 50
            im3 = axes[5].imshow(dist_display, cmap='viridis')
            axes[5].set_title('Shore Distance')
            axes[5].axis('off')
            
            # Overlay
            overlay = result['rgb'].copy()
            for plume in result['plume_info']:
                y1, x1, y2, x2 = plume['bbox']
                overlay[int(y1):int(y2), int(x1):int(x2), 0] = 255
            axes[6].imshow(overlay)
            axes[6].set_title('RGB + Detections')
            axes[6].axis('off')
            
            # Ocean temps histogram
            ocean_temps = result['thermal'][result['masks']['ocean']]
            if len(ocean_temps) > 0:
                axes[7].hist(ocean_temps, bins=50, alpha=0.7, color='blue')
                axes[7].axvline(result['threshold'], color='red', linestyle='--', 
                              label=f'Threshold: {result["threshold"]:.1f}°C')
                axes[7].axvline(result['ocean_median'], color='green', linestyle='--',
                              label=f'Median: {result["ocean_median"]:.1f}°C')
                axes[7].set_xlabel('Temperature (°C)')
                axes[7].set_ylabel('Count')
                axes[7].set_title('Ocean Temperature Distribution')
                axes[7].legend(fontsize=8)
                axes[7].grid(True, alpha=0.3)
            
            # Thermal profile along shore
            if result['shoreline_thin'].any():
                shore_points = np.argwhere(result['shoreline_thin'])
                if len(shore_points) > 0:
                    # Sample along shoreline
                    sample_idx = np.linspace(0, len(shore_points)-1, 100, dtype=int)
                    shore_temps = []
                    for idx in sample_idx:
                        y, x = shore_points[idx]
                        # Get temperature in small neighborhood
                        y_min = max(0, y-2)
                        y_max = min(result['thermal'].shape[0], y+3)
                        x_min = max(0, x-2)
                        x_max = min(result['thermal'].shape[1], x+3)
                        local_temp = np.nanmean(result['thermal'][y_min:y_max, x_min:x_max])
                        shore_temps.append(local_temp)
                    
                    axes[8].plot(shore_temps, 'b-', alpha=0.7)
                    axes[8].axhline(result['threshold'], color='red', linestyle='--')
                    axes[8].set_xlabel('Position along shore')
                    axes[8].set_ylabel('Temperature (°C)')
                    axes[8].set_title('Temperature Along Shoreline')
                    axes[8].grid(True, alpha=0.3)
            
            # Plume details
            if result['plume_info']:
                plume_text = "Detected Plumes:\n\n"
                for i, plume in enumerate(result['plume_info'][:5]):  # Show first 5
                    plume_text += f"Plume {i+1}:\n"
                    plume_text += f"  Area: {plume['area_pixels']} px\n"
                    plume_text += f"  Shore dist: {plume['min_shore_distance']:.1f} px\n"
                    plume_text += f"  Eccentricity: {plume['eccentricity']:.2f}\n\n"
            else:
                plume_text = "No SGD plumes detected"
            
            axes[9].text(0.05, 0.95, plume_text, transform=axes[9].transAxes,
                        fontsize=8, verticalalignment='top')
            axes[9].axis('off')
            
            # Stats
            stats_text = f"Parameters:\n"
            stats_text += f"  Temp threshold: {current['temp_threshold']:.1f}°C\n"
            stats_text += f"  Min area: {current['min_area']} px\n\n"
            
            if result['characteristics']:
                stats_text += f"SGD Statistics:\n"
                stats_text += f"  Mean temp: {result['characteristics']['mean_temp']:.1f}°C\n"
                stats_text += f"  Anomaly: {result['characteristics']['temp_anomaly']:.1f}°C\n"
                stats_text += f"  Total area: {result['characteristics']['area_m2']:.1f} m²\n"
            
            axes[10].text(0.05, 0.5, stats_text, transform=axes[10].transAxes,
                         fontsize=9, verticalalignment='center')
            axes[10].axis('off')
            
            # Enhanced thermal for ocean
            ocean_only = result['thermal'].copy()
            ocean_only[~result['masks']['ocean']] = np.nan
            ocean_min = np.nanmin(ocean_only)
            ocean_max = np.nanmax(ocean_only)
            ocean_enhanced = (ocean_only - ocean_min) / (ocean_max - ocean_min + 1e-10)
            
            im4 = axes[11].imshow(ocean_enhanced, cmap='plasma')
            axes[11].contour(result['sgd_plumes'], colors='lime', linewidths=2)
            axes[11].set_title('Enhanced Ocean + SGD')
            axes[11].axis('off')
            
            fig.canvas.draw_idle()
        
        # Add controls
        from matplotlib.widgets import Slider, Button
        
        ax_frame = plt.axes([0.1, 0.02, 0.4, 0.02])
        frame_slider = Slider(ax_frame, 'Frame', 0, len(frame_numbers)-1, 
                            valinit=0, valstep=1)
        
        ax_temp = plt.axes([0.55, 0.02, 0.2, 0.02])
        temp_slider = Slider(ax_temp, 'Temp Δ(°C)', 0.5, 3.0, 
                           valinit=1.0, valstep=0.1)
        
        ax_area = plt.axes([0.8, 0.02, 0.15, 0.02])
        area_slider = Slider(ax_area, 'Min Area', 10, 200, 
                           valinit=50, valstep=10)
        
        def update_frame(val):
            current['frame_idx'] = int(frame_slider.val)
            update_display()
        
        def update_temp(val):
            current['temp_threshold'] = temp_slider.val
            update_display()
        
        def update_area(val):
            current['min_area'] = int(area_slider.val)
            update_display()
        
        frame_slider.on_changed(update_frame)
        temp_slider.on_changed(update_temp)
        area_slider.on_changed(update_area)
        
        # Initial display
        update_display()
        
        plt.show()


def main():
    """Main entry point"""
    print("Submarine Groundwater Discharge (SGD) Detector")
    print("=" * 50)
    
    analyzer = SGDAnalyzer()
    
    print("\nOptions:")
    print("1. Interactive viewer (adjust parameters)")
    print("2. Batch process frames (export to GIS)")
    print("3. Quick test (single frame)")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == '1':
        print("\nLaunching interactive SGD detector...")
        print("Adjust sliders to tune detection parameters")
        analyzer.interactive_viewer()
        
    elif choice == '2':
        print("\nBatch processing for SGD detection...")
        results = analyzer.batch_process(export_gis=True)
        print(f"\nProcessed {len(results)} frames")
        print("Results saved to sgd_output/")
        
    else:
        # Quick test
        frame = int(input("Enter frame number (default 248): ") or "248")
        detector = SGDDetector()
        result = detector.process_frame(frame)
        
        print(f"\nFrame {frame} Results:")
        print(f"  SGD plumes detected: {len(result['plume_info'])}")
        if result['characteristics']:
            print(f"  Temperature anomaly: {result['characteristics']['temp_anomaly']:.1f}°C")
            print(f"  Total SGD area: {result['characteristics']['area_m2']:.1f} m²")
        
        fig = detector.visualize_sgd_detection(result)
        plt.show()


if __name__ == "__main__":
    main()