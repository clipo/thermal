#!/usr/bin/env python3
"""
Interactive tool for collecting training data for segmentation.
Click on image regions to label them as ocean, land, rock, or wave.
Trains a classifier to automatically segment based on your labels.
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
    """Interactive tool for training segmentation classifier"""
    
    def __init__(self, base_path="data/100MEDIA",
                 model_file="segmentation_model.pkl",
                 training_file="segmentation_training_data.json"):
        """Initialize trainer
        
        Args:
            base_path: Path to data directory
            model_file: Path to save/load ML model
            training_file: Path to save/load training data
        """
        self.base_path = Path(base_path)
        self.training_file = training_file
        self.model_file = model_file
        
        # Find available frames
        self.frames = []
        for f in sorted(self.base_path.glob("MAX_*.JPG"))[:100]:
            num = int(f.stem.split('_')[1])
            if (self.base_path / f"IRX_{num:04d}.irg").exists():
                self.frames.append(num)
        
        if not self.frames:
            raise FileNotFoundError("No paired frames found!")
        
        print(f"Found {len(self.frames)} frames")
        
        # Initialize
        self.current_frame_idx = 0
        self.current_frame = self.frames[0]
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
        self.load_frame(self.current_frame)
        
        # Setup GUI
        self.setup_gui()
        self.update_display()
        
        # Show initial instruction
        self.show_status("👆 Click on image to label pixels. Need 100+ samples per class", 'blue')
    
    def load_training_data(self):
        """Load existing training data if available"""
        if Path(self.training_file).exists():
            with open(self.training_file, 'r') as f:
                data = json.load(f)
                print(f"Loaded {len(data)} existing training samples")
                return data
        return []
    
    def save_training_data(self):
        """Save training data to file"""
        with open(self.training_file, 'w') as f:
            json.dump(self.training_data, f, indent=2)
        print(f"Saved {len(self.training_data)} training samples")
    
    def load_frame(self, frame_number):
        """Load and process a frame"""
        # Load RGB image
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        self.rgb_full = np.array(PILImage.open(rgb_path))
        
        # Extract thermal FOV region (center 70%)
        self.thermal_fov_ratio = 0.7
        h, w = self.rgb_full.shape[:2]
        crop_h = int(h * self.thermal_fov_ratio)
        crop_w = int(w * self.thermal_fov_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        self.rgb_aligned = self.rgb_full[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Resize for display
        img_pil = PILImage.fromarray(self.rgb_aligned)
        self.rgb_display = np.array(img_pil.resize((640, 512), PILImage.Resampling.BILINEAR))
        
        # Compute features
        self.compute_features()
        
        self.current_frame = frame_number
        
        # Initialize label mask for this frame
        self.label_mask = np.zeros(self.rgb_display.shape[:2], dtype=int)
        self.has_labels = np.zeros(self.rgb_display.shape[:2], dtype=bool)
    
    def compute_features(self):
        """Compute color features for each pixel"""
        # Convert to different color spaces
        self.hsv = color.rgb2hsv(self.rgb_display)
        self.lab = color.rgb2lab(self.rgb_display)
        
        # RGB channels
        r = self.rgb_display[:,:,0].astype(float)
        g = self.rgb_display[:,:,1].astype(float)
        b = self.rgb_display[:,:,2].astype(float)
        
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
        
        # Store features
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
        
        if x < 0 or x >= self.rgb_display.shape[1] or y < 0 or y >= self.rgb_display.shape[0]:
            return
        
        # Extract features for this location
        features = self.extract_pixel_features(x, y)
        
        # Add to training data
        sample = {
            'frame': self.current_frame,
            'x': x,
            'y': y,
            'label': self.current_label,
            'features': features
        }
        
        self.training_data.append(sample)
        
        # Mark this region as labeled
        y_min = max(0, y - self.sample_radius)
        y_max = min(self.rgb_display.shape[0], y + self.sample_radius + 1)
        x_min = max(0, x - self.sample_radius)
        x_max = min(self.rgb_display.shape[1], x + self.sample_radius + 1)
        
        label_idx = self.classes.index(self.current_label)
        self.label_mask[y_min:y_max, x_min:x_max] = label_idx
        self.has_labels[y_min:y_max, x_min:x_max] = True
        
        print(f"Added {self.current_label} sample at ({x}, {y}) - Total samples: {len(self.training_data)}")
        
        # Update display
        self.update_display()
        
        # Auto-save every 10 samples
        if len(self.training_data) % 10 == 0:
            self.save_training_data()
    
    def update_display(self):
        """Update the display"""
        # Clear axes
        self.ax_image.clear()
        self.ax_labels.clear()
        
        # Show image
        self.ax_image.imshow(self.rgb_display)
        self.ax_image.set_title(f'Frame {self.current_frame} - Click to label as: {self.current_label.upper()}')
        self.ax_image.axis('off')
        
        # Create label overlay
        overlay = np.zeros((*self.label_mask.shape, 4))
        for i, class_name in enumerate(self.classes):
            mask = (self.label_mask == i) & self.has_labels
            color = self.class_colors[class_name]
            overlay[mask] = color + [0.5]  # Add alpha
        
        # Show labels
        self.ax_labels.imshow(self.rgb_display)
        self.ax_labels.imshow(overlay)
        self.ax_labels.set_title('Labeled Regions')
        self.ax_labels.axis('off')
        
        # Statistics
        self.update_stats()
        
        plt.draw()
    
    def update_stats(self):
        """Update statistics display"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        # Count samples per class
        class_counts = {c: 0 for c in self.classes}
        for sample in self.training_data:
            class_counts[sample['label']] += 1
        
        stats_text = f"Training Statistics\n"
        stats_text += "=" * 30 + "\n"
        stats_text += f"Total samples: {len(self.training_data)}\n\n"
        stats_text += "Samples per class:\n"
        for class_name in self.classes:
            stats_text += f"  {class_name:8s}: {class_counts[class_name]:4d}\n"
        
        stats_text += f"\nCurrent frame: {self.current_frame}\n"
        stats_text += f"Frame {self.current_frame_idx + 1}/{len(self.frames)}"
        
        self.ax_stats.text(0.1, 0.9, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, family='monospace', va='top')
    
    def set_label(self, label):
        """Set current label for clicking"""
        self.current_label = label
        print(f"Now labeling: {label}")
        self.update_display()
    
    def next_frame(self, event):
        """Go to next frame"""
        if self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.load_frame(self.frames[self.current_frame_idx])
            self.update_display()
    
    def prev_frame(self, event):
        """Go to previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.load_frame(self.frames[self.current_frame_idx])
            self.update_display()
    
    def train_classifier(self, event):
        """Train a classifier on collected data"""
        if len(self.training_data) < 40:
            print(f"Need at least 40 samples, have {len(self.training_data)}")
            self.show_status(f"❌ Need at least 40 samples, have {len(self.training_data)}", 'red')
            return
        
        # Show training is starting
        self.show_status("🔄 Training classifier... Please wait", 'orange')
        plt.pause(0.1)  # Force update
        
        print("\nTraining classifier...")
        import time
        start_time = time.time()
        
        # Prepare data
        X = np.array([s['features'] for s in self.training_data])
        y = np.array([self.classes.index(s['label']) for s in self.training_data])
        
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
        print(classification_report(y_test, y_pred, 
                                   target_names=self.classes))
        
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
            if Path(self.model_file).exists():
                with open(self.model_file, 'rb') as f:
                    self.classifier = pickle.load(f)
                print("Loaded existing model")
            else:
                print("No trained model available")
                return
        
        print("Testing classifier on current frame...")
        
        # Prepare features for all pixels
        h, w = self.rgb_display.shape[:2]
        predictions = np.zeros((h, w), dtype=int)
        
        # Process in chunks for efficiency
        chunk_size = 10
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
    
    def clear_frame_labels(self, event):
        """Clear labels for current frame"""
        # Remove samples for current frame
        self.training_data = [s for s in self.training_data if s['frame'] != self.current_frame]
        self.save_training_data()
        
        # Reset display
        self.load_frame(self.current_frame)
        self.update_display()
    
    def show_status(self, message, color='black'):
        """Show status message in the interface"""
        if not hasattr(self, 'status_text'):
            # Create status text area if it doesn't exist
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
        print(f"Cleared labels for frame {self.current_frame}")
    
    def save_and_exit(self, event):
        """Save model and training data, then close the window"""
        print("\n" + "="*50)
        print("SAVING AND EXITING")
        print("="*50)
        
        # Save training data
        self.save_training_data()
        
        # Check if we have a trained model to save
        if hasattr(self, 'classifier'):
            # Save the model
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.classifier, f)
            print(f"✓ Model saved to: {self.model_file}")
        else:
            print("⚠ No trained model to save. Train the model first (press 'T')")
            print("  Continuing without saving model...")
        
        print("✓ Training data saved")
        print("\nClosing trainer and continuing to SGD detection...")
        print("="*50)
        
        # Close the matplotlib window
        plt.close(self.fig)
    
    def setup_gui(self):
        """Setup the GUI"""
        self.fig = plt.figure(figsize=(18, 10))
        
        # Image displays
        self.ax_image = plt.subplot(2, 3, 1)
        self.ax_labels = plt.subplot(2, 3, 2)
        self.ax_pred = plt.subplot(2, 3, 3)
        self.ax_stats = plt.subplot(2, 3, 4)
        
        # Radio buttons for label selection
        ax_radio = plt.axes([0.05, 0.3, 0.1, 0.15])
        self.radio = RadioButtons(ax_radio, self.classes)
        self.radio.on_clicked(self.set_label)
        
        # Navigation buttons
        ax_prev = plt.axes([0.3, 0.02, 0.08, 0.04])
        ax_next = plt.axes([0.39, 0.02, 0.08, 0.04])
        self.btn_prev = Button(ax_prev, 'Prev')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        
        # Action buttons
        ax_train = plt.axes([0.5, 0.02, 0.08, 0.04])
        ax_test = plt.axes([0.59, 0.02, 0.08, 0.04])
        ax_clear = plt.axes([0.68, 0.02, 0.08, 0.04])
        ax_save = plt.axes([0.77, 0.02, 0.08, 0.04])
        
        self.btn_train = Button(ax_train, 'Train')
        self.btn_test = Button(ax_test, 'Test')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_save = Button(ax_save, 'Save & Continue')
        
        self.btn_train.on_clicked(self.train_classifier)
        self.btn_test.on_clicked(self.test_classifier)
        self.btn_clear.on_clicked(self.clear_frame_labels)
        self.btn_save.on_clicked(self.save_and_exit)
        
        # Connect click handler
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Instructions
        ax_inst = plt.subplot(2, 3, 6)
        ax_inst.axis('off')
        instructions = """
INSTRUCTIONS:
1. Select label type (ocean/land/rock/wave)
2. Click on image regions to label them
3. Navigate frames with Prev/Next
4. Press 'Train' when you have enough labels
5. Press 'Test' to see ML segmentation
6. Press 'Save & Continue' to proceed to detection

Tips:
- Label diverse examples
- Include edge cases
- Label from multiple frames
- Need 40+ samples to train
- IMPORTANT: Press 'Train' before 'Save & Continue'
        """
        ax_inst.text(0.1, 0.9, instructions, transform=ax_inst.transAxes,
                    fontsize=9, family='monospace', va='top')
    
    def run(self):
        """Run the trainer"""
        print("\nSegmentation Trainer")
        print("=" * 50)
        print("Click on regions to label them for training")
        print("Build a dataset then train the classifier")
        plt.show()

def main():
    """Main entry point with command-line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive Segmentation Training Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model and training data
  python segmentation_trainer.py
  
  # Use different data directory
  python segmentation_trainer.py --data data/flight2
  
  # Create a new model for specific conditions
  python segmentation_trainer.py --model rocky_shore_model.pkl --training rocky_shore_data.json
  
  # Continue training an existing model with different data
  python segmentation_trainer.py --model sunrise_model.pkl --training sunrise_data.json --data data/survey3
  
Controls:
  - Left click: Label as Ocean
  - Right click: Label as Land
  - Middle click: Label as Rock
  - Shift+click: Label as Wave
  - T key: Train classifier
  - S key: Save model
  - C key: Clear current frame
  - Space: Next frame
        """
    )
    
    parser.add_argument('--model', type=str, default='segmentation_model.pkl',
                       help='Path to save/load ML model (default: segmentation_model.pkl)')
    parser.add_argument('--training', type=str, default='segmentation_training_data.json',
                       help='Path to save/load training data (default: segmentation_training_data.json)')
    parser.add_argument('--data', type=str, default='data/100MEDIA',
                       help='Path to data directory (default: data/100MEDIA)')
    
    args = parser.parse_args()
    
    print(f"Model file: {args.model}")
    print(f"Training data: {args.training}")
    print(f"Data directory: {args.data}")
    print()
    
    trainer = SegmentationTrainer(
        base_path=args.data,
        model_file=args.model,
        training_file=args.training
    )
    trainer.run()

if __name__ == "__main__":
    main()