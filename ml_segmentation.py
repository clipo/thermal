#!/usr/bin/env python3
"""
ML-based segmentation using trained classifier.
This module can be integrated into the SGD detector for better segmentation.
"""

import numpy as np
import pickle
from pathlib import Path
from skimage import color, morphology
from PIL import Image as PILImage

class MLSegmenter:
    """Machine learning based segmentation"""
    
    def __init__(self, model_path="segmentation_model.pkl"):
        """Initialize with trained model"""
        self.model_path = Path(model_path)
        self.classifier = None
        self.sample_radius = 5
        
        # Classes
        self.classes = ['ocean', 'land', 'rock', 'wave']
        
        # Load model if available
        if self.model_path.exists():
            self.load_model()
    
    def load_model(self):
        """Load trained classifier"""
        try:
            with open(self.model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            print(f"Loaded ML segmentation model from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def compute_features(self, rgb_image):
        """Compute features for an RGB image"""
        # Ensure uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # Convert to different color spaces
        hsv = color.rgb2hsv(rgb_image)
        lab = color.rgb2lab(rgb_image)
        
        # RGB channels
        r = rgb_image[:,:,0].astype(float)
        g = rgb_image[:,:,1].astype(float)
        b = rgb_image[:,:,2].astype(float)
        
        # Compute additional features
        intensity = (r + g + b) / 3
        
        # Blue dominance
        blue_dominance = np.zeros_like(b)
        mask = (r > 0) | (g > 0)
        blue_dominance[mask] = b[mask] / (np.maximum(r[mask], g[mask]) + 1)
        
        # Color variance
        max_channel = np.maximum(r, np.maximum(g, b))
        min_channel = np.minimum(r, np.minimum(g, b))
        color_range = max_channel - min_channel
        
        # Stack features
        features = np.stack([
            r / 255.0,                 # Red
            g / 255.0,                 # Green
            b / 255.0,                 # Blue
            hsv[:,:,0],                # Hue
            hsv[:,:,1],                # Saturation
            hsv[:,:,2],                # Value
            lab[:,:,0] / 100.0,        # L
            lab[:,:,1] / 128.0,        # a
            lab[:,:,2] / 128.0,        # b
            intensity / 255.0,         # Intensity
            blue_dominance,            # Blue dominance
            color_range / 255.0,       # Color variance
        ], axis=2)
        
        return features
    
    def extract_pixel_features(self, features, x, y):
        """Extract features for a pixel and its neighborhood"""
        h, w = features.shape[:2]
        
        # Get neighborhood
        x_min = max(0, x - self.sample_radius)
        x_max = min(w, x + self.sample_radius + 1)
        y_min = max(0, y - self.sample_radius)
        y_max = min(h, y + self.sample_radius + 1)
        
        # Extract features from neighborhood
        neighborhood_features = features[y_min:y_max, x_min:x_max, :]
        
        # Compute statistics
        pixel_features = []
        for i in range(neighborhood_features.shape[2]):
            channel = neighborhood_features[:,:,i]
            pixel_features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])
        
        return pixel_features
    
    def segment(self, rgb_image, chunk_size=20):
        """
        Segment an RGB image using the trained classifier.
        
        Returns:
            dict with 'ocean', 'land', 'wave' masks
        """
        if self.classifier is None:
            print("No trained model available. Use segmentation_trainer.py to create one.")
            return None
        
        # Compute features
        features = self.compute_features(rgb_image)
        h, w = rgb_image.shape[:2]
        
        # Initialize predictions
        predictions = np.zeros((h, w), dtype=int)
        
        # Process in chunks for efficiency
        total_pixels = h * w
        processed = 0
        
        print(f"Segmenting {h}x{w} image...")
        
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                y_end = min(y + chunk_size, h)
                x_end = min(x + chunk_size, w)
                
                # Extract features for chunk
                chunk_features = []
                positions = []
                
                for py in range(y, y_end):
                    for px in range(x, x_end):
                        pixel_features = self.extract_pixel_features(features, px, py)
                        chunk_features.append(pixel_features)
                        positions.append((py, px))
                
                # Predict
                if chunk_features:
                    chunk_pred = self.classifier.predict(chunk_features)
                    
                    # Store predictions
                    for (py, px), pred in zip(positions, chunk_pred):
                        predictions[py, px] = pred
                
                processed += len(chunk_features)
                if processed % 10000 == 0:
                    print(f"  Processed {processed}/{total_pixels} pixels ({100*processed/total_pixels:.1f}%)")
        
        # Create masks
        ocean_mask = predictions == self.classes.index('ocean')
        land_mask = predictions == self.classes.index('land')
        rock_mask = predictions == self.classes.index('rock')
        wave_mask = predictions == self.classes.index('wave')
        
        # Combine rock with land
        land_mask = land_mask | rock_mask
        
        # Morphological cleanup
        ocean_mask = morphology.remove_small_objects(ocean_mask, min_size=100)
        land_mask = morphology.remove_small_objects(land_mask, min_size=100)
        wave_mask = morphology.remove_small_objects(wave_mask, min_size=25)
        
        # Fill holes
        ocean_mask = morphology.remove_small_holes(ocean_mask, area_threshold=200)
        land_mask = morphology.remove_small_holes(land_mask, area_threshold=200)
        
        print("Segmentation complete!")
        
        return {
            'ocean': ocean_mask,
            'land': land_mask,
            'waves': wave_mask
        }
    
    def segment_fast(self, rgb_image, stride=2):
        """
        Faster segmentation by sampling pixels with stride.
        Less accurate but much faster.
        """
        if self.classifier is None:
            print("No trained model available.")
            return None
        
        # Compute features
        features = self.compute_features(rgb_image)
        h, w = rgb_image.shape[:2]
        
        # Sample pixels with stride
        sampled_predictions = np.zeros((h//stride + 1, w//stride + 1), dtype=int)
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                pixel_features = self.extract_pixel_features(features, x, y)
                pred = self.classifier.predict([pixel_features])[0]
                sampled_predictions[y//stride, x//stride] = pred
        
        # Upscale predictions
        from scipy.ndimage import zoom
        predictions = zoom(sampled_predictions, stride, order=0)
        predictions = predictions[:h, :w]
        
        # Create masks
        ocean_mask = predictions == self.classes.index('ocean')
        land_mask = (predictions == self.classes.index('land')) | \
                   (predictions == self.classes.index('rock'))
        wave_mask = predictions == self.classes.index('wave')
        
        # Cleanup
        ocean_mask = morphology.remove_small_objects(ocean_mask, min_size=100)
        land_mask = morphology.remove_small_objects(land_mask, min_size=100)
        wave_mask = morphology.remove_small_objects(wave_mask, min_size=25)
        
        return {
            'ocean': ocean_mask,
            'land': land_mask,
            'waves': wave_mask
        }

def test_ml_segmentation():
    """Test the ML segmentation on a sample image"""
    segmenter = MLSegmenter()
    
    if segmenter.classifier is None:
        print("No model found. Train one first using segmentation_trainer.py")
        return
    
    # Load a test image
    test_frame = 248
    rgb_path = Path(f"data/100MEDIA/MAX_{test_frame:04d}.JPG")
    
    if not rgb_path.exists():
        print(f"Test image not found: {rgb_path}")
        return
    
    # Load and crop to thermal FOV
    rgb_full = np.array(PILImage.open(rgb_path))
    h, w = rgb_full.shape[:2]
    crop_h = int(h * 0.7)
    crop_w = int(w * 0.7)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    rgb_cropped = rgb_full[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    # Resize for testing
    img_pil = PILImage.fromarray(rgb_cropped)
    rgb_test = np.array(img_pil.resize((640, 512), PILImage.Resampling.BILINEAR))
    
    # Segment
    print(f"Testing ML segmentation on frame {test_frame}...")
    masks = segmenter.segment_fast(rgb_test, stride=3)
    
    if masks:
        # Display results
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(rgb_test)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Create colored segmentation
        segmentation = np.zeros((*masks['ocean'].shape, 3))
        segmentation[masks['ocean']] = [0, 0.3, 1]
        segmentation[masks['land']] = [0, 0.7, 0]
        segmentation[masks['waves']] = [1, 1, 0.5]
        
        axes[1].imshow(segmentation)
        axes[1].set_title('ML Segmentation')
        axes[1].axis('off')
        
        # Overlay
        overlay = rgb_test / 255.0
        overlay[masks['ocean']] = overlay[masks['ocean']] * 0.5 + [0, 0.15, 0.5]
        overlay[masks['waves']] = overlay[masks['waves']] * 0.5 + [0.5, 0.5, 0.25]
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        total = masks['ocean'].size
        print(f"\nSegmentation statistics:")
        print(f"  Ocean: {100*masks['ocean'].sum()/total:.1f}%")
        print(f"  Land:  {100*masks['land'].sum()/total:.1f}%")
        print(f"  Waves: {100*masks['waves'].sum()/total:.1f}%")

if __name__ == "__main__":
    test_ml_segmentation()