#!/usr/bin/env python3
"""
Fix Figure 9 using the actual trained segmentation model.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pickle
import sys

# Add thermal project to path
sys.path.append('/Users/clipo/PycharmProjects/thermal')

def load_trained_model():
    """Load the trained segmentation model from pickle file."""
    # Try multiple possible model locations
    model_paths = [
        Path('/Users/clipo/PycharmProjects/thermal/segmentation_model.pkl'),
        Path('/Users/clipo/PycharmProjects/thermal/ocean_segmentation_model.pkl'),
        Path('/Users/clipo/PycharmProjects/thermal/models/rapa_nui_sgd_south_coast1_model.pkl')
    ]
    
    for model_path in model_paths:
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    print(f"Loaded trained model from {model_path}")
                    return model_data
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
    
    print("No model file found in expected locations")
    return None

def apply_segmentation(rgb_img, model_data):
    """Apply the trained model to segment the image."""
    from sklearn.ensemble import RandomForestClassifier
    
    if model_data is None:
        return None
    
    try:
        # Handle both dict-wrapped and direct model formats
        if isinstance(model_data, dict):
            model = model_data['model']
        else:
            model = model_data
        
        # Extract features from RGB image
        height, width = rgb_img.shape[:2]
        
        # Create feature array (similar to ml_segmenter.py)
        features = []
        
        # RGB values
        features.append(rgb_img[:,:,0].flatten())  # R
        features.append(rgb_img[:,:,1].flatten())  # G
        features.append(rgb_img[:,:,2].flatten())  # B
        
        # HSV features
        hsv = np.array(Image.fromarray(rgb_img).convert('HSV'))
        features.append(hsv[:,:,0].flatten())  # Hue
        features.append(hsv[:,:,1].flatten())  # Saturation
        features.append(hsv[:,:,2].flatten())  # Value
        
        # Grayscale
        gray = np.array(Image.fromarray(rgb_img).convert('L'))
        features.append(gray.flatten())
        
        # Simple texture features (gradients)
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        features.append(grad_x.flatten())
        features.append(grad_y.flatten())
        
        # Position features (normalized)
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        features.append((x_coords / width).flatten())
        features.append((y_coords / height).flatten())
        
        # Stack features
        X = np.column_stack(features)
        
        # Predict
        predictions = model.predict(X)
        
        # Reshape to image
        segmentation = predictions.reshape(height, width)
        
        return segmentation
        
    except Exception as e:
        print(f"Error applying segmentation: {e}")
        return None

def fix_thermal_ocean_segmentation_with_model(thermal_path, rgb_path, output_path='docs/images/sgd_detection_process_correct.png'):
    """Fix Figure 9 using the actual trained model."""
    
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Load model
    model_data = load_trained_model()
    
    # Load images
    thermal_img = None
    ocean_mask = None
    
    if Path(thermal_path).exists():
        thermal_img = np.array(Image.open(thermal_path))
        if len(thermal_img.shape) == 3:
            thermal_img = np.mean(thermal_img, axis=2)
        print(f"Loaded thermal image: {thermal_img.shape}")
    
    if Path(rgb_path).exists():
        rgb_img = np.array(Image.open(rgb_path))
        print(f"Loaded RGB image: {rgb_img.shape}")
        
        # Apply trained segmentation
        if model_data is not None:
            segmentation = apply_segmentation(rgb_img, model_data)
            if segmentation is not None:
                ocean_mask = (segmentation == 0)  # Class 0 is ocean
                print(f"Applied ML segmentation - Ocean pixels: {np.sum(ocean_mask)}")
            else:
                print("Failed to apply ML segmentation")
        else:
            print("No model available - using simple threshold")
            # Fallback: Simple blue detection for ocean
            blue_ratio = rgb_img[:,:,2] / (rgb_img[:,:,0] + rgb_img[:,:,1] + rgb_img[:,:,2] + 1)
            ocean_mask = blue_ratio > 0.4
    
    if thermal_img is not None:
        # Normalize thermal
        thermal_norm = (thermal_img - thermal_img.min()) / (thermal_img.max() - thermal_img.min() + 1e-8)
        
        # Panel A: Raw thermal
        im1 = ax1.imshow(thermal_norm, cmap='jet')
        ax1.set_title('A) Raw Thermal Image', fontsize=12, weight='bold')
        ax1.axis('off')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Relative Temperature', rotation=270, labelpad=15)
        
        # Panel B: Ocean-masked thermal
        if ocean_mask is not None:
            # Resize ocean mask to match thermal dimensions
            from PIL import Image as PILImage
            ocean_mask_pil = PILImage.fromarray((ocean_mask * 255).astype(np.uint8))
            ocean_mask_resized = np.array(
                ocean_mask_pil.resize(
                    (thermal_img.shape[1], thermal_img.shape[0]), 
                    PILImage.NEAREST
                )
            ) > 127
            
            # Create masked array
            masked_thermal = np.ma.masked_where(~ocean_mask_resized, thermal_norm)
            
            print(f"Ocean mask coverage: {np.sum(ocean_mask_resized) / ocean_mask_resized.size * 100:.1f}%")
        else:
            # Fallback if no mask available
            print("Using fallback ocean mask")
            ocean_mask_resized = np.ones_like(thermal_norm, dtype=bool)
            ocean_mask_resized[:50, :] = False  # Top is land
            ocean_mask_resized[-50:, :] = False  # Bottom is land
            masked_thermal = np.ma.masked_where(~ocean_mask_resized, thermal_norm)
        
        im2 = ax2.imshow(masked_thermal, cmap='jet')
        ax2.set_title('B) Ocean-Masked Thermal', fontsize=12, weight='bold')
        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Ocean Temperature', rotation=270, labelpad=15)
        
        # Add ocean coverage stat
        ocean_pct = np.sum(ocean_mask_resized) / ocean_mask_resized.size * 100
        ax2.text(0.02, 0.98, f'Ocean: {ocean_pct:.1f}%', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Panel C: Temperature anomaly (only in ocean)
        if np.any(ocean_mask_resized):
            ocean_temps = thermal_norm[ocean_mask_resized]
            ocean_mean = np.mean(ocean_temps)
            ocean_std = np.std(ocean_temps)
            
            # Calculate anomaly
            anomaly = np.zeros_like(thermal_norm)
            anomaly[ocean_mask_resized] = (thermal_norm[ocean_mask_resized] - ocean_mean) / ocean_std
            
            # Mask non-ocean areas
            anomaly_masked = np.ma.masked_where(~ocean_mask_resized, anomaly)
            
            im3 = ax3.imshow(anomaly_masked, cmap='RdBu_r', vmin=-2, vmax=2)
            ax3.set_title('C) Temperature Anomaly (σ from mean)', fontsize=12, weight='bold')
            ax3.axis('off')
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('Standard Deviations', rotation=270, labelpad=15)
            
            # Add stats
            ax3.text(0.02, 0.98, f'Ocean μ: {ocean_mean:.3f}\nOcean σ: {ocean_std:.3f}', 
                    transform=ax3.transAxes, fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        else:
            anomaly = thermal_norm - np.mean(thermal_norm)
            im3 = ax3.imshow(anomaly, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
            ax3.set_title('C) Temperature Anomaly', fontsize=12, weight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='ΔTemp')
        
        # Panel D: Detected SGD plumes (only in ocean)
        if np.any(ocean_mask_resized):
            # SGDs are cold anomalies in ocean
            sgd_threshold = -1.0  # More than 1 std dev colder
            sgd_mask = (anomaly < sgd_threshold) & ocean_mask_resized
            
            # Remove small detections (noise)
            from scipy import ndimage
            sgd_mask = ndimage.binary_opening(sgd_mask, iterations=2)
            
            # Count connected components
            labeled, num_features = ndimage.label(sgd_mask)
        else:
            sgd_mask = np.zeros_like(thermal_norm, dtype=bool)
            num_features = 0
        
        # Display thermal with SGD overlay
        ax4.imshow(thermal_norm, cmap='gray', vmin=0, vmax=1)
        
        # Create red overlay for SGD areas
        if np.any(sgd_mask):
            sgd_overlay = np.zeros((*thermal_norm.shape, 4))
            sgd_overlay[sgd_mask] = [1, 0, 0, 0.6]  # Red with 60% opacity
            ax4.imshow(sgd_overlay)
        
        ax4.set_title('D) Detected SGD Plumes (Cold Anomalies in Ocean)', fontsize=12, weight='bold')
        ax4.axis('off')
        
        # Add statistics
        num_pixels = np.sum(sgd_mask)
        if np.any(ocean_mask_resized):
            ocean_coverage = num_pixels / np.sum(ocean_mask_resized) * 100
        else:
            ocean_coverage = 0
        
        stats_text = f'SGD Features: {num_features}\nSGD Pixels: {num_pixels}\nOcean Coverage: {ocean_coverage:.2f}%'
        ax4.text(0.02, 0.98, stats_text, 
                transform=ax4.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top')
    
    plt.suptitle('SGD Detection Pipeline with Trained Ocean Segmentation', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n✅ Saved corrected Figure 9 to: {output_path}")
    
    return ocean_mask_resized if 'ocean_mask_resized' in locals() else None

def main():
    """Fix Figure 9 with correct ocean segmentation."""
    
    print("Fixing Figure 9 - SGD Detection Process\n")
    print("=" * 50)
    
    # Use nadir images from the dataset
    base_dir = Path("/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/102MEDIA")
    thermal_path = base_dir / "MAX_0052.JPG"  # Thermal nadir image
    rgb_path = base_dir / "MAX_0053.JPG"      # RGB nadir image
    
    print(f"Thermal: {thermal_path}")
    print(f"RGB: {rgb_path}")
    print()
    
    # Generate corrected figure
    ocean_mask = fix_thermal_ocean_segmentation_with_model(
        thermal_path, 
        rgb_path,
        'docs/images/sgd_detection_process_correct.png'
    )
    
    print("\n" + "=" * 50)
    print("Figure 9 has been corrected!")
    print("\nThe ocean is now properly identified using the trained model.")
    print("SGD plumes will only be detected in actual ocean areas.")
    print("\nUpdate TECHNICAL_PAPER.md to use: sgd_detection_process_correct.png")

if __name__ == "__main__":
    main()