#!/usr/bin/env python3
"""
Optimized FAST ML segmentation using vectorized operations.
Much faster than pixel-by-pixel processing.
"""

import numpy as np
import pickle
from pathlib import Path
from skimage import color, morphology
from scipy.ndimage import uniform_filter, zoom
import time

class FastMLSegmenter:
    """Optimized fast ML segmentation"""
    
    def __init__(self, model_path="segmentation_model.pkl"):
        """Initialize with trained model"""
        self.model_path = Path(model_path)
        self.classifier = None
        self.classes = ['ocean', 'land', 'rock', 'wave']
        
        if self.model_path.exists():
            self.load_model()
    
    def load_model(self):
        """Load trained classifier"""
        try:
            with open(self.model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            print(f"Loaded fast ML model")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def compute_features_vectorized(self, rgb_image, window_size=5):
        """
        Compute features using vectorized operations (MUCH faster).
        Uses convolution instead of pixel-by-pixel processing.
        """
        # Ensure uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        h, w = rgb_image.shape[:2]
        
        # Basic color spaces
        hsv = color.rgb2hsv(rgb_image)
        lab = color.rgb2lab(rgb_image)
        
        # RGB channels
        r = rgb_image[:,:,0].astype(float) / 255.0
        g = rgb_image[:,:,1].astype(float) / 255.0
        b = rgb_image[:,:,2].astype(float) / 255.0
        
        # Fast neighborhood statistics using uniform filter
        half_window = window_size // 2
        
        # Create feature layers
        features = []
        
        # Color features (direct values)
        features.append(r)  # Red
        features.append(g)  # Green
        features.append(b)  # Blue
        features.append(hsv[:,:,0])  # Hue
        features.append(hsv[:,:,1])  # Saturation
        features.append(hsv[:,:,2])  # Value
        features.append(lab[:,:,0] / 100.0)  # L
        features.append(lab[:,:,1] / 128.0)  # a
        features.append(lab[:,:,2] / 128.0)  # b
        
        # Derived features
        intensity = (r + g + b) / 3
        features.append(intensity)
        
        # Blue dominance
        blue_dominance = np.zeros_like(b)
        mask = (r > 0) | (g > 0)
        blue_dominance[mask] = b[mask] / (np.maximum(r[mask], g[mask]) + 0.001)
        features.append(blue_dominance)
        
        # Color range
        max_channel = np.maximum(r, np.maximum(g, b))
        min_channel = np.minimum(r, np.minimum(g, b))
        color_range = max_channel - min_channel
        features.append(color_range)
        
        # Compute neighborhood statistics using fast filters
        feature_array = []
        for feat in features:
            # Mean
            feat_mean = uniform_filter(feat, size=window_size)
            feature_array.append(feat_mean)
            
            # Std (approximation using range)
            feat_max = uniform_filter(feat, size=window_size, mode='constant')
            feat_min = uniform_filter(feat, size=window_size, mode='constant')
            feat_range = feat_max - feat_min
            feature_array.append(feat_range * 0.3)  # Approximate std
            
            # Min and max (use original for speed)
            feature_array.append(feat)
            feature_array.append(feat)
        
        # Stack all features
        return np.stack(feature_array, axis=2)
    
    def segment_ultra_fast(self, rgb_image, downsample=4):
        """
        Ultra-fast segmentation by downsampling.
        Process at lower resolution then upscale.
        """
        if self.classifier is None:
            print("No model available")
            return None
        
        h_orig, w_orig = rgb_image.shape[:2]
        
        # Downsample image
        from PIL import Image as PILImage
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        img_pil = PILImage.fromarray(rgb_image)
        h_small = h_orig // downsample
        w_small = w_orig // downsample
        rgb_small = np.array(img_pil.resize((w_small, h_small), PILImage.Resampling.BILINEAR))
        
        # Compute features on small image
        features = self.compute_features_vectorized(rgb_small, window_size=3)
        
        # Reshape for sklearn
        n_features = features.shape[2]
        features_flat = features.reshape(-1, n_features)
        
        # Predict in batches
        batch_size = 10000
        predictions_flat = np.zeros(features_flat.shape[0], dtype=int)
        
        for i in range(0, features_flat.shape[0], batch_size):
            end = min(i + batch_size, features_flat.shape[0])
            predictions_flat[i:end] = self.classifier.predict(features_flat[i:end])
        
        # Reshape predictions
        predictions_small = predictions_flat.reshape(h_small, w_small)
        
        # Upscale predictions to original size
        # Use resize to ensure exact dimensions
        from PIL import Image as PILImage
        pred_img = PILImage.fromarray(predictions_small.astype(np.uint8))
        pred_resized = pred_img.resize((w_orig, h_orig), PILImage.Resampling.NEAREST)
        predictions = np.array(pred_resized)
        
        # Create masks
        ocean_mask = predictions == self.classes.index('ocean')
        land_mask = (predictions == self.classes.index('land')) | \
                   (predictions == self.classes.index('rock'))
        wave_mask = predictions == self.classes.index('wave')
        
        # Ensure masks have correct dimensions
        assert ocean_mask.shape == (h_orig, w_orig), f"Mask shape mismatch: {ocean_mask.shape} vs {(h_orig, w_orig)}"
        
        # Quick morphological cleanup
        ocean_mask = morphology.remove_small_objects(ocean_mask, min_size=100)
        land_mask = morphology.remove_small_objects(land_mask, min_size=100)

        # Keep only the largest contiguous ocean area
        # This prevents small landlocked areas from being misclassified as ocean
        from skimage import measure
        ocean_labels = measure.label(ocean_mask, connectivity=2)
        if ocean_labels.max() > 0:
            # Find the largest connected component
            unique_labels, counts = np.unique(ocean_labels[ocean_labels > 0], return_counts=True)
            if len(unique_labels) > 0:
                largest_count = np.max(counts)
                # Only keep if the largest area is significant (>5% of image)
                # This prevents keeping small misclassified patches when drone is over land
                total_pixels = ocean_mask.size
                if largest_count > total_pixels * 0.05:  # At least 5% of image
                    largest_label = unique_labels[np.argmax(counts)]
                    ocean_mask = (ocean_labels == largest_label)
                else:
                    # No significant ocean area - drone is over land
                    ocean_mask = np.zeros_like(ocean_mask, dtype=bool)

                # Update land mask to include the removed small "ocean" areas
                land_mask = ~(ocean_mask | wave_mask)

        return {
            'ocean': ocean_mask,
            'land': land_mask,
            'waves': wave_mask
        }

def test_fast_segmentation():
    """Test the fast segmentation"""
    from PIL import Image as PILImage
    
    print("Testing Fast ML Segmentation")
    print("=" * 40)
    
    # Load test image
    rgb_path = Path("data/100MEDIA/MAX_0248.JPG")
    rgb_full = np.array(PILImage.open(rgb_path))
    
    # Crop and resize
    h, w = rgb_full.shape[:2]
    crop_h = int(h * 0.7)
    crop_w = int(w * 0.7)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    rgb_cropped = rgb_full[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    img_pil = PILImage.fromarray(rgb_cropped)
    rgb_test = np.array(img_pil.resize((640, 512), PILImage.Resampling.BILINEAR))
    
    # Test fast segmenter
    segmenter = FastMLSegmenter()
    
    if segmenter.classifier:
        print(f"Image size: {rgb_test.shape}")
        
        for downsample in [2, 3, 4]:
            print(f"\nDownsample factor: {downsample}")
            start = time.time()
            masks = segmenter.segment_ultra_fast(rgb_test, downsample=downsample)
            elapsed = time.time() - start
            print(f"  Time: {elapsed:.3f} seconds")
            
            if masks:
                total = masks['ocean'].size
                print(f"  Ocean: {100*masks['ocean'].sum()/total:.1f}%")
                print(f"  Land:  {100*masks['land'].sum()/total:.1f}%")
    else:
        print("No model found")

if __name__ == "__main__":
    test_fast_segmentation()