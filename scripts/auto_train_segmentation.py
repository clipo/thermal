#!/usr/bin/env python3
"""
Automated segmentation training module for batch processing
Automatically labels and trains a segmentation model based on image statistics
"""

import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from skimage import color
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class AutoSegmentationTrainer:
    """Automated segmentation training using heuristics"""
    
    def __init__(self, data_dir, model_file, training_file=None, sample_frames=10, verbose=True):
        """
        Initialize automated trainer
        
        Args:
            data_dir: Directory containing MAX_*.JPG and IRX_*.irg files
            model_file: Output model filename
            training_file: Optional training data filename for saving labels
            sample_frames: Number of frames to sample for training
            verbose: Show progress messages
        """
        self.data_dir = Path(data_dir)
        self.model_file = model_file
        self.training_file = training_file
        self.sample_frames = sample_frames
        self.verbose = verbose
        
        # Find available frames
        self.frames = []
        for f in sorted(self.data_dir.glob("MAX_*.JPG")):
            num = int(f.stem.split('_')[1])
            if (self.data_dir / f"IRX_{num:04d}.irg").exists():
                self.frames.append(num)
        
        if not self.frames:
            raise FileNotFoundError(f"No paired frames found in {data_dir}")
        
        if self.verbose:
            print(f"Found {len(self.frames)} frames for training")
        
        # Classes for segmentation
        self.classes = ['ocean', 'land', 'rock', 'wave']
        
        # Training data storage
        self.training_data = []
        
    def extract_features(self, rgb_array, x, y, radius=5):
        """Extract features for a pixel and its neighborhood"""
        h, w = rgb_array.shape[:2]
        
        # Get neighborhood bounds
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        
        # Get neighborhood pixels
        neighborhood = rgb_array[y_min:y_max, x_min:x_max]
        
        # Convert to different color spaces
        hsv = color.rgb2hsv(neighborhood)
        lab = color.rgb2lab(neighborhood)
        
        # RGB channels
        r = neighborhood[:,:,0].astype(float)
        g = neighborhood[:,:,1].astype(float)
        b = neighborhood[:,:,2].astype(float)
        
        # Compute features
        intensity = (r + g + b) / 3
        
        # Blue dominance (useful for water detection)
        blue_dominance = np.zeros_like(b)
        mask = (r > 0) | (g > 0)
        blue_dominance[mask] = b[mask] / (np.maximum(r[mask], g[mask]) + 1)
        
        # Color variance
        max_channel = np.maximum(r, np.maximum(g, b))
        min_channel = np.minimum(r, np.minimum(g, b))
        color_range = max_channel - min_channel
        
        # Extract statistics
        features = []
        
        # RGB statistics
        for channel in [r, g, b]:
            features.extend([
                np.mean(channel) / 255.0,
                np.std(channel) / 255.0,
                np.min(channel) / 255.0,
                np.max(channel) / 255.0
            ])
        
        # HSV statistics
        for i in range(3):
            channel = hsv[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])
        
        # LAB statistics
        for i in range(3):
            channel = lab[:,:,i]
            # Normalize LAB values
            if i == 0:
                channel = channel / 100.0  # L channel
            else:
                channel = channel / 128.0  # a and b channels
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])
        
        # Additional features
        features.extend([
            np.mean(intensity) / 255.0,
            np.std(intensity) / 255.0,
            np.mean(blue_dominance),
            np.std(blue_dominance),
            np.mean(color_range) / 255.0,
            np.std(color_range) / 255.0
        ])
        
        return features
    
    def classify_pixel_heuristic(self, rgb_array, x, y):
        """
        Classify a pixel using heuristics based on color
        
        Returns:
            str: Class label ('ocean', 'land', 'rock', 'wave')
        """
        # Get pixel color
        r, g, b = rgb_array[y, x]
        
        # Convert to HSV for better classification
        hsv_pixel = color.rgb2hsv(rgb_array[y:y+1, x:x+1])[0, 0]
        hue = hsv_pixel[0]
        saturation = hsv_pixel[1]
        value = hsv_pixel[2]
        
        # Calculate some useful metrics
        intensity = (r + g + b) / 3
        blue_dominance = b / (max(r, g) + 1) if (r > 0 or g > 0) else 0
        
        # Classification rules based on typical appearance
        
        # Wave/foam detection (very bright, low saturation)
        if value > 0.8 and saturation < 0.3:
            return 'wave'
        
        # Ocean detection (blue-dominant)
        if blue_dominance > 1.3 and b > 50:
            # Deep water is darker blue
            if intensity < 100:
                return 'ocean'
            # Shallow water or waves might be brighter
            elif intensity > 180:
                return 'wave'
            else:
                return 'ocean'
        
        # Rock detection (gray, low saturation)
        if saturation < 0.2 and 50 < intensity < 150:
            return 'rock'
        
        # Land detection (everything else - vegetation, sand, etc.)
        # Vegetation tends to be green
        if g > r and g > b:
            return 'land'
        
        # Sandy/brown areas
        if r > b and saturation > 0.2:
            return 'land'
        
        # Default to land for anything else
        return 'land'
    
    def process_frame(self, frame_number):
        """Process a single frame to generate training data"""
        # Load RGB image
        rgb_path = self.data_dir / f"MAX_{frame_number:04d}.JPG"
        rgb_full = np.array(PILImage.open(rgb_path))
        
        # Extract thermal FOV region (center 70%)
        thermal_fov_ratio = 0.7
        h, w = rgb_full.shape[:2]
        crop_h = int(h * thermal_fov_ratio)
        crop_w = int(w * thermal_fov_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        rgb_aligned = rgb_full[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Resize for processing
        img_pil = PILImage.fromarray(rgb_aligned)
        rgb_display = np.array(img_pil.resize((640, 512), PILImage.Resampling.BILINEAR))
        
        # Sample pixels from the image
        h, w = rgb_display.shape[:2]
        
        # Create a grid of sample points
        sample_spacing = 20  # Sample every 20 pixels
        
        samples_added = 0
        for y in range(10, h - 10, sample_spacing):
            for x in range(10, w - 10, sample_spacing):
                # Classify using heuristics
                label = self.classify_pixel_heuristic(rgb_display, x, y)
                
                # Extract features
                features = self.extract_features(rgb_display, x, y)
                
                # Add to training data
                sample = {
                    'frame': frame_number,
                    'x': x,
                    'y': y,
                    'label': label,
                    'features': features
                }
                
                self.training_data.append(sample)
                samples_added += 1
        
        if self.verbose:
            print(f"  Frame {frame_number}: Added {samples_added} training samples")
        
        return samples_added
    
    def train(self):
        """Automatically generate training data and train the model"""
        if self.verbose:
            print("\nAutomated Segmentation Training")
            print("=" * 50)
            print(f"Sampling {self.sample_frames} frames for training...")
        
        # Sample frames evenly across the dataset
        frame_indices = np.linspace(0, len(self.frames) - 1, 
                                   min(self.sample_frames, len(self.frames)), 
                                   dtype=int)
        
        # Process selected frames
        total_samples = 0
        for idx in frame_indices:
            frame_num = self.frames[idx]
            samples = self.process_frame(frame_num)
            total_samples += samples
        
        if self.verbose:
            print(f"\nTotal training samples collected: {total_samples}")
            
            # Count samples per class
            class_counts = {c: 0 for c in self.classes}
            for sample in self.training_data:
                class_counts[sample['label']] += 1
            
            print("\nSamples per class:")
            for class_name in self.classes:
                print(f"  {class_name:8s}: {class_counts[class_name]:6d}")
        
        # Train the model
        if len(self.training_data) < 40:
            raise ValueError(f"Insufficient training data: {len(self.training_data)} samples")
        
        if self.verbose:
            print("\nTraining Random Forest classifier...")
        
        # Prepare data
        X = np.array([s['features'] for s in self.training_data])
        y = np.array([self.classes.index(s['label']) for s in self.training_data])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.classifier.score(X_train, y_train)
        test_score = self.classifier.score(X_test, y_test)
        
        if self.verbose:
            print(f"  Training accuracy: {train_score:.2%}")
            print(f"  Testing accuracy:  {test_score:.2%}")
        
        # Save model
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        if self.verbose:
            print(f"\n✓ Model saved to: {self.model_file}")
        
        # Save training data if requested
        if self.training_file:
            # Convert numpy arrays to lists for JSON serialization
            training_data_json = []
            for sample in self.training_data:
                sample_json = sample.copy()
                sample_json['features'] = [float(f) for f in sample['features']]
                training_data_json.append(sample_json)
            
            with open(self.training_file, 'w') as f:
                json.dump(training_data_json, f, indent=2)
            
            if self.verbose:
                print(f"✓ Training data saved to: {self.training_file}")
        
        return {
            'total_samples': len(self.training_data),
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'class_counts': class_counts if self.verbose else None
        }


def main():
    """Standalone training script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automated segmentation model training'
    )
    
    parser.add_argument('--data', required=True,
                       help='Directory containing images')
    parser.add_argument('--output', required=True,
                       help='Output model filename')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of frames to sample (default: 10)')
    parser.add_argument('--save-training', 
                       help='Save training data to JSON file')
    
    args = parser.parse_args()
    
    trainer = AutoSegmentationTrainer(
        data_dir=args.data,
        model_file=args.output,
        training_file=args.save_training,
        sample_frames=args.samples,
        verbose=True
    )
    
    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"Model accuracy: {stats['test_accuracy']:.2%}")


if __name__ == "__main__":
    main()