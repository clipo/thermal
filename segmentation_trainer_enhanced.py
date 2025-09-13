#!/usr/bin/env python3
"""
Enhanced segmentation trainer with better frame sampling and area-based model naming.
This replaces the original segmentation_trainer.py with improvements.
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from pathlib import Path
from PIL import Image as PILImage
from skimage import color
import json
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import random
import warnings
warnings.filterwarnings('ignore')

class SegmentationTrainer:
    """Interactive tool for training segmentation classifier with enhanced sampling"""
    
    def __init__(self, base_path, 
                 model_file=None,
                 training_file=None,
                 sampling='distributed',
                 increment=25,
                 max_frames=20):
        """Initialize trainer with improved sampling
        
        Args:
            base_path: Path to data directory or parent of XXXMEDIA directories
            model_file: Path to save/load ML model (auto-generated if None)
            training_file: Path to save/load training data (auto-generated if None)
            sampling: 'increment' (every Nth), 'random', or 'distributed' (evenly spaced)
            increment: Frame increment for 'increment' sampling
            max_frames: Maximum frames to use for training
        """
        self.base_path = Path(base_path)
        
        # Get area name for model naming
        self.area_name = self.get_area_name(base_path)
        
        # Find frames with better sampling
        self.frame_paths = self.find_training_frames(sampling, increment, max_frames)
        
        if not self.frame_paths:
            raise FileNotFoundError(f"No frames found in {base_path}")
        
        # Auto-generate model paths based on area name
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        if model_file is None:
            self.model_file = models_dir / f"{self.area_name}_segmentation.pkl"
        else:
            self.model_file = Path(model_file)
            
        if training_file is None:
            self.training_file = models_dir / f"{self.area_name}_training.json"
        else:
            self.training_file = Path(training_file)
        
        print(f"\n{'='*60}")
        print(f"SEGMENTATION TRAINING - {self.area_name.upper()}")
        print(f"{'='*60}")
        print(f"  Area: {self.area_name}")
        print(f"  Frames: {len(self.frame_paths)} ({sampling} sampling)")
        print(f"  Model: {self.model_file.name}")
        print(f"  Training data: {self.training_file.name}")
        
        # Show frame distribution
        frame_nums = self.get_frame_numbers()
        if len(frame_nums) <= 10:
            print(f"  Frame numbers: {frame_nums}")
        else:
            print(f"  Frame numbers: {frame_nums[:5]}... and {len(frame_nums)-5} more")
        print(f"{'='*60}\n")
        
        # Initialize
        self.current_frame_idx = 0
        self.current_frame_path = self.frame_paths[0]
        self.current_label = 'ocean'  # Default label
        
        # Training data storage
        self.training_data = self.load_training_data()
        
        # Classes
        self.classes = ['ocean', 'land', 'rock', 'wave']
        self.class_colors = {
            'ocean': [0, 0.3, 1],
            'land': [0, 0.7, 0],
            'rock': [0.5, 0.3, 0.1],
            'wave': [1, 1, 0.5]
        }
        
        # Sampling parameters
        self.sample_radius = 5  # Pixels around click point
        
        # Load initial frame
        self.load_frame(self.current_frame_path)
        
        # Setup GUI
        self.setup_gui()
        self.update_display()
        
        # Show initial instruction
        self.show_status("👆 Click on image to label pixels. Need 100+ samples per class", 'blue')
    
    def get_area_name(self, data_path):
        """Get area name from directory for model naming"""
        data_path = Path(data_path)
        
        # If it's a XXXMEDIA directory, go up one level
        if data_path.name.endswith('MEDIA'):
            area_path = data_path.parent
        else:
            area_path = data_path
        
        # Clean the area name
        area_name = area_path.name.lower()
        
        # Remove common prefixes
        area_name = area_name.replace('flight ', '').replace('flight_', '')
        area_name = area_name.replace(' - ', '_').replace('-', '_')
        area_name = area_name.replace(' ', '_')
        
        # Remove non-alphanumeric except underscore
        area_name = ''.join(c if c.isalnum() or c == '_' else '' for c in area_name)
        
        return area_name if area_name else 'unnamed_area'
    
    def find_training_frames(self, sampling='distributed', increment=25, max_frames=20):
        """Find frames with better sampling strategy"""
        frames = []
        
        # Check for XXXMEDIA subdirectories
        media_dirs = sorted([d for d in self.base_path.iterdir() 
                           if d.is_dir() and d.name.endswith('MEDIA')])
        
        if media_dirs:
            print(f"Found {len(media_dirs)} MEDIA directories:")
            # Collect frames from all MEDIA directories
            for media_dir in media_dirs:
                dir_frames = sorted(media_dir.glob("MAX_*.JPG"))
                if dir_frames:
                    frames.extend(dir_frames)
                    print(f"  {media_dir.name}: {len(dir_frames)} frames")
        else:
            # Single directory
            frames = sorted(self.base_path.glob("MAX_*.JPG"))
            print(f"Found {len(frames)} frames in {self.base_path.name}")
        
        if not frames:
            return []
        
        print(f"Total frames available: {len(frames)}")
        
        # Sample frames based on strategy
        if sampling == 'increment':
            sampled = frames[::increment]
            print(f"Using increment sampling (every {increment} frames): {len(sampled)} frames")
        elif sampling == 'random':
            num_samples = min(max_frames, len(frames))
            sampled = random.sample(frames, num_samples)
            print(f"Using random sampling: {num_samples} frames")
        elif sampling == 'distributed':
            num_samples = min(max_frames, len(frames))
            if num_samples == len(frames):
                sampled = frames
            else:
                # Evenly distributed indices
                indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)
                sampled = [frames[i] for i in indices]
            print(f"Using distributed sampling: {len(sampled)} frames evenly spaced")
        else:
            sampled = frames[:max_frames]
        
        # Limit to max_frames
        if len(sampled) > max_frames:
            sampled = sampled[:max_frames]
            print(f"Limited to {max_frames} frames maximum")
        
        return sampled
    
    def get_frame_numbers(self):
        """Extract frame numbers from paths for display"""
        numbers = []
        for path in self.frame_paths:
            try:
                # Extract number from MAX_XXXX.JPG
                num = int(path.stem.split('_')[1])
                numbers.append(num)
            except:
                pass
        return numbers
    
    def load_training_data(self):
        """Load existing training data if available"""
        if self.training_file.exists():
            with open(self.training_file, 'r') as f:
                data = json.load(f)
                print(f"Loaded {len(data)} existing training samples")
                return data
        return []
    
    def save_training_data(self):
        """Save training data to file"""
        with open(self.training_file, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        print(f"Saved {len(self.training_data)} training samples to {self.training_file.name}")
    
    def load_frame(self, frame_path):
        """Load and process a frame"""
        frame_path = Path(frame_path)
        
        # Load RGB image
        if not frame_path.exists():
            print(f"Warning: Frame not found: {frame_path}")
            return
        
        rgb_img = PILImage.open(frame_path)
        self.rgb_image = np.array(rgb_img)
        
        # Create display version (smaller for GUI)
        max_size = 800
        h, w = self.rgb_image.shape[:2]
        if w > max_size:
            scale = max_size / w
            new_size = (int(w * scale), int(h * scale))
            rgb_img_resized = rgb_img.resize(new_size, PILImage.LANCZOS)
            self.rgb_display = np.array(rgb_img_resized)
        else:
            self.rgb_display = self.rgb_image.copy()
        
        # Get frame identifier
        self.current_frame = frame_path.stem
        
        # Compute features
        self.compute_features()
        
        # Track labeled pixels
        self.has_labels = np.zeros(self.rgb_display.shape[:2], dtype=bool)
        
        # Mark existing labels
        for sample in self.training_data:
            if sample.get('frame') == self.current_frame:
                x, y = sample['pixel']
                y_min = max(0, y - self.sample_radius)
                y_max = min(self.rgb_display.shape[0], y + self.sample_radius + 1)
                x_min = max(0, x - self.sample_radius)
                x_max = min(self.rgb_display.shape[1], x + self.sample_radius + 1)
                self.has_labels[y_min:y_max, x_min:x_max] = True
    
    def compute_features(self):
        """Compute features for the current frame"""
        # Convert to different color spaces
        self.hsv = color.rgb2hsv(self.rgb_display)
        self.lab = color.rgb2lab(self.rgb_display)
        
        # Extract channels
        r = self.rgb_display[:,:,0]
        g = self.rgb_display[:,:,1]
        b = self.rgb_display[:,:,2]
        
        # Compute additional features
        intensity = np.mean(self.rgb_display, axis=2)
        blue_dominance = (b - np.maximum(r, g)) / 255.0
        color_range = (np.max([r, g, b], axis=0) - np.min([r, g, b], axis=0))
        
        # Stack features
        self.features = np.stack([
            r / 255.0,                      # Red
            g / 255.0,                      # Green
            b / 255.0,                      # Blue
            self.hsv[:,:,0],                # Hue
            self.hsv[:,:,1],                # Saturation
            self.hsv[:,:,2],                # Value
            self.lab[:,:,0] / 100.0,        # L
            self.lab[:,:,1] / 128.0,        # a
            self.lab[:,:,2] / 128.0,        # b
            intensity / 255.0,               # Intensity
            blue_dominance,                  # Blue dominance
            color_range / 255.0,            # Color variance
        ], axis=2)
        
        print(f"Computed {self.features.shape[2]} features per pixel")
    
    def extract_pixel_features(self, x, y):
        """Extract features for a pixel and its neighborhood"""
        h, w = self.rgb_display.shape[:2]
        
        # Get neighborhood
        x_min = max(0, x - self.sample_radius)
        x_max = min(w, x + self.sample_radius + 1)
        y_min = max(0, y - self.sample_radius)
        y_max = min(h, y + self.sample_radius + 1)
        
        # Extract features from neighborhood
        neighborhood_features = self.features[y_min:y_max, x_min:x_max, :]
        
        # Compute statistics
        features = []
        for i in range(neighborhood_features.shape[2]):
            channel = neighborhood_features[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])
        
        return features
    
    def on_click(self, event):
        """Handle mouse clicks for labeling"""
        if event.inaxes != self.ax_image:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        # Extract features
        features = self.extract_pixel_features(x, y)
        
        # Store training sample
        sample = {
            'frame': self.current_frame,
            'pixel': [x, y],
            'label': self.current_label,
            'features': features,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_data.append(sample)
        
        # Mark this region as labeled
        y_min = max(0, y - self.sample_radius)
        y_max = min(self.rgb_display.shape[0], y + self.sample_radius + 1)
        x_min = max(0, x - self.sample_radius)
        x_max = min(self.rgb_display.shape[1], x + self.sample_radius + 1)
        
        self.has_labels[y_min:y_max, x_min:x_max] = True
        
        print(f"Added {self.current_label} sample at ({x}, {y}) - Total: {len(self.training_data)}")
        
        # Update display
        self.update_display()
        
        # Auto-save every 10 samples
        if len(self.training_data) % 10 == 0:
            self.save_training_data()
    
    def update_display(self):
        """Update the display"""
        # Clear axes
        self.ax_image.clear()
        
        # Show image
        self.ax_image.imshow(self.rgb_display)
        self.ax_image.set_title(f'Frame: {self.current_frame} ({self.current_frame_idx+1}/{len(self.frame_paths)})')
        self.ax_image.axis('off')
        
        # Overlay labeled regions
        overlay = np.zeros((*self.rgb_display.shape[:2], 4))
        
        # Show existing labels for this frame
        for sample in self.training_data:
            if sample.get('frame') == self.current_frame:
                x, y = sample['pixel']
                label = sample['label']
                color = self.class_colors[label]
                
                # Draw a circle at the labeled point
                circle = plt.Circle((x, y), self.sample_radius, 
                                   color=color, alpha=0.5)
                self.ax_image.add_patch(circle)
        
        # Update statistics
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Count samples per class
        class_counts = {c: 0 for c in self.classes}
        for sample in self.training_data:
            class_counts[sample['label']] += 1
        
        stats_text = f"Training Statistics ({self.area_name})\n"
        stats_text += "=" * 30 + "\n"
        stats_text += f"Total samples: {len(self.training_data)}\n\n"
        stats_text += "Samples per class:\n"
        for class_name in self.classes:
            count = class_counts[class_name]
            status = "✓" if count >= 100 else f"need {100-count} more"
            stats_text += f"  {class_name:8s}: {count:4d} {status}\n"
        
        stats_text += f"\nCurrent frame: {self.current_frame}\n"
        stats_text += f"Frame {self.current_frame_idx + 1} of {len(self.frame_paths)}"
        
        self.ax_stats.text(0.1, 0.9, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top', family='monospace')
        
        plt.draw()
    
    def setup_gui(self):
        """Setup the GUI"""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle(f'Segmentation Training - {self.area_name}', fontsize=14, weight='bold')
        
        # Image display
        self.ax_image = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        
        # Class selector
        self.ax_radio = plt.subplot2grid((3, 4), (0, 2))
        self.radio = RadioButtons(self.ax_radio, self.classes)
        self.radio.on_clicked(self.set_label)
        
        # Statistics
        self.ax_stats = plt.subplot2grid((3, 4), (0, 3), rowspan=2)
        
        # Buttons
        self.ax_prev = plt.subplot2grid((3, 4), (2, 0))
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_frame)
        
        self.ax_next = plt.subplot2grid((3, 4), (2, 1))
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self.next_frame)
        
        self.ax_train = plt.subplot2grid((3, 4), (2, 2))
        self.btn_train = Button(self.ax_train, 'Train', color='lightgreen')
        self.btn_train.on_clicked(self.train_classifier)
        
        self.ax_test = plt.subplot2grid((3, 4), (2, 3))
        self.btn_test = Button(self.ax_test, 'Test', color='lightblue')
        self.btn_test.on_clicked(self.test_classifier)
        
        # Prediction display
        self.ax_pred = plt.subplot2grid((3, 4), (1, 2), colspan=2)
        self.ax_pred.axis('off')
        
        # Connect click handler
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
    
    def set_label(self, label):
        """Set current label"""
        self.current_label = label
        print(f"Current label: {label}")
    
    def next_frame(self, event):
        """Go to next frame"""
        if self.current_frame_idx < len(self.frame_paths) - 1:
            self.current_frame_idx += 1
            self.current_frame_path = self.frame_paths[self.current_frame_idx]
            self.load_frame(self.current_frame_path)
            self.update_display()
    
    def prev_frame(self, event):
        """Go to previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.current_frame_path = self.frame_paths[self.current_frame_idx]
            self.load_frame(self.current_frame_path)
            self.update_display()
    
    def train_classifier(self, event):
        """Train classifier with progress indication"""
        # Check minimum samples
        min_samples = 100  # Require 100+ per class for good training
        class_counts = {c: 0 for c in self.classes}
        for sample in self.training_data:
            class_counts[sample['label']] += 1
        
        under_sampled = [c for c in self.classes if class_counts[c] < min_samples]
        if under_sampled:
            msg = f"❌ Need 100+ samples per class. Missing: {', '.join(under_sampled)}"
            print(msg)
            self.show_status(msg, 'red')
            return
        
        # Clear test panel
        self.ax_pred.clear()
        self.ax_pred.axis('off')
        
        # Show training is starting
        self.show_status("🔄 Training classifier... Please wait", 'orange')
        plt.pause(0.1)
        
        print("\nTraining classifier...")
        import time
        start_time = time.time()
        
        # Prepare data
        X = np.array([s['features'] for s in self.training_data])
        y = np.array([self.classes.index(s['label']) for s in self.training_data])
        
        # Show preparation status
        self.show_status(f"📊 Training on {len(X)} samples...", 'orange')
        plt.pause(0.1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
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
        y_pred = self.classifier.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        # Show results
        elapsed = time.time() - start_time
        self.show_status(f"✅ Training complete! Accuracy: {accuracy:.1%} (took {elapsed:.1f}s)", 'green')
        
        print("\nClassifier Performance:")
        print(classification_report(y_test, y_pred, target_names=self.classes))
        
        # Save model
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Model saved to {self.model_file}")
        
        # Test on current frame
        self.test_classifier(None)
    
    def test_classifier(self, event):
        """Test classifier on current frame"""
        if not hasattr(self, 'classifier'):
            # Try to load existing model
            if self.model_file.exists():
                with open(self.model_file, 'rb') as f:
                    self.classifier = pickle.load(f)
                print(f"Loaded existing model from {self.model_file.name}")
                self.show_status("📂 Loaded existing model", 'green')
            else:
                print("No trained model available")
                self.show_status("❌ No classifier trained yet! Click 'Train' first", 'red')
                return
        
        self.show_status("🔍 Testing segmentation...", 'orange')
        plt.pause(0.1)
        
        print("\nTesting segmentation on current frame...")
        
        # Prepare features for entire image
        h, w = self.rgb_display.shape[:2]
        predictions = np.zeros((h, w), dtype=int)
        
        # Process in chunks for speed
        chunk_size = 50
        for y in range(0, h, chunk_size):
            for x in range(0, w, chunk_size):
                y_end = min(y + chunk_size, h)
                x_end = min(x + chunk_size, w)
                
                # Extract features for chunk
                chunk_features = []
                for py in range(y, y_end):
                    for px in range(x, x_end):
                        features = self.extract_pixel_features(px, py)
                        chunk_features.append(features)
                
                # Predict
                if chunk_features:
                    chunk_pred = self.classifier.predict(chunk_features)
                    
                    # Store predictions
                    idx = 0
                    for py in range(y, y_end):
                        for px in range(x, x_end):
                            predictions[py, px] = chunk_pred[idx]
                            idx += 1
        
        # Display predictions
        self.ax_pred.clear()
        
        # Create colored segmentation
        segmentation = np.zeros((h, w, 3))
        for i, class_name in enumerate(self.classes):
            mask = predictions == i
            segmentation[mask] = self.class_colors[class_name]
        
        # Clear and update display
        self.ax_pred.clear()
        self.ax_pred.imshow(segmentation)
        self.ax_pred.set_title('ML Segmentation', fontsize=10)
        self.ax_pred.axis('off')
        
        # Calculate statistics
        unique, counts = np.unique(predictions, return_counts=True)
        ocean_pct = 0
        if 0 in unique:  # Ocean is class 0
            idx = np.where(unique == 0)[0][0]
            ocean_pct = counts[idx] / predictions.size * 100
        
        # Show status
        self.show_status(f"✅ Segmentation complete! Ocean: {ocean_pct:.1f}%", 'green')
        
        plt.draw()
        print("Segmentation complete!")
        print(f"Ocean coverage: {ocean_pct:.1f}%")
    
    def show_status(self, message, color='black'):
        """Show status message in the interface"""
        if not hasattr(self, 'status_text'):
            # Create status text area
            self.status_text = self.fig.text(0.5, 0.02, '', 
                                            ha='center', va='bottom',
                                            fontsize=11, weight='bold',
                                            bbox=dict(boxstyle='round', 
                                                    facecolor='white', 
                                                    edgecolor=color,
                                                    linewidth=2))
        
        self.status_text.set_text(message)
        self.status_text.set_bbox(dict(boxstyle='round', 
                                      facecolor='white', 
                                      edgecolor=color,
                                      linewidth=2))
        plt.draw()
    
    def run(self):
        """Run the trainer"""
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Interactive segmentation training tool')
    parser.add_argument('--data', type=str, default='data/100MEDIA',
                       help='Path to data directory or parent of XXXMEDIA dirs')
    parser.add_argument('--sampling', type=str, default='distributed',
                       choices=['increment', 'random', 'distributed'],
                       help='Frame sampling strategy')
    parser.add_argument('--increment', type=int, default=25,
                       help='Frame increment for increment sampling')
    parser.add_argument('--max-frames', type=int, default=20,
                       help='Maximum frames to use for training')
    parser.add_argument('--model', type=str, default=None,
                       help='Model file path (auto-generated if not specified)')
    parser.add_argument('--training', type=str, default=None,
                       help='Training data file path (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    trainer = SegmentationTrainer(
        args.data,
        model_file=args.model,
        training_file=args.training,
        sampling=args.sampling,
        increment=args.increment,
        max_frames=args.max_frames
    )
    trainer.run()

if __name__ == "__main__":
    main()