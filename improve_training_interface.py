#!/usr/bin/env python3
"""
Improvements to the segmentation training interface.
This file contains the enhanced functions to add to segmentation_trainer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import Rectangle
import time

def enhanced_train_classifier(self, event):
    """Enhanced training with progress indicator"""
    if len(self.training_data) < 40:
        # Show error in interface
        self.show_status(f"❌ Need at least 40 samples, have {len(self.training_data)}", 'red')
        return
    
    # Clear any previous test visualization
    if hasattr(self, 'ax_test'):
        self.ax_test.clear()
        self.ax_test.axis('off')
        plt.draw()
    
    # Show training is starting
    self.show_status("🔄 Training classifier... Please wait", 'orange')
    plt.pause(0.1)  # Force update
    
    print("\nTraining classifier...")
    start_time = time.time()
    
    # Prepare data
    X = np.array([s['features'] for s in self.training_data])
    y = np.array([self.classes.index(s['label']) for s in self.training_data])
    
    # Show data preparation status
    self.show_status(f"📊 Preparing {len(X)} samples...", 'orange')
    plt.pause(0.1)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train classifier with progress
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    self.show_status(f"🌲 Training Random Forest on {len(X_train)} samples...", 'orange')
    plt.pause(0.1)
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Test
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Show results
    elapsed = time.time() - start_time
    self.show_status(f"✅ Training complete! Accuracy: {accuracy:.1%} (took {elapsed:.1f}s)", 'green')
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=self.classes))
    
    # Save model
    import pickle
    with open(self.model_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {self.model_file}")
    
    self.classifier = clf
    
    # Auto-test after training
    self.test_segmentation(None)

def enhanced_test_segmentation(self, event):
    """Enhanced test with proper visualization"""
    if not hasattr(self, 'classifier') or self.classifier is None:
        self.show_status("❌ No classifier trained yet! Click 'Train' first", 'red')
        return
    
    self.show_status("🔍 Testing segmentation...", 'orange')
    plt.pause(0.1)
    
    print("\nTesting segmentation on current frame...")
    
    # Prepare features for entire image
    h, w = self.rgb_display.shape[:2]
    all_features = []
    
    # Show progress for feature extraction
    total_pixels = h * w
    chunk_size = 50  # Process in chunks for progress
    
    for y in range(0, h, chunk_size):
        for x in range(0, w, chunk_size):
            # Update progress
            progress = (y * w + x) / total_pixels * 100
            if progress % 10 < 1:  # Update every ~10%
                self.show_status(f"🔍 Extracting features... {progress:.0f}%", 'orange')
                plt.pause(0.01)
            
            y_end = min(y + chunk_size, h)
            x_end = min(x + chunk_size, w)
            
            for yy in range(y, y_end):
                for xx in range(x, x_end):
                    features = self.extract_pixel_features(xx, yy)
                    all_features.append(features)
    
    X = np.array(all_features)
    
    # Predict
    self.show_status("🎯 Predicting classes...", 'orange')
    plt.pause(0.1)
    
    predictions = self.classifier.predict(X)
    segmentation = predictions.reshape(h, w)
    
    # Create colored segmentation
    seg_colored = np.zeros((h, w, 3))
    for i, class_name in enumerate(self.classes):
        mask = segmentation == i
        color = self.class_colors[class_name]
        seg_colored[mask] = color
    
    # Display in test panel with proper scaling
    self.ax_test.clear()
    self.ax_test.imshow(seg_colored)
    self.ax_test.set_title('Segmentation Result', fontsize=10)
    self.ax_test.axis('off')
    
    # Add legend to test panel
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=self.class_colors[c], label=c) 
                      for c in self.classes]
    self.ax_test.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Calculate statistics
    unique, counts = np.unique(segmentation, return_counts=True)
    total = len(predictions)
    
    stats_text = "Segmentation Results:\n"
    for i, class_name in enumerate(self.classes):
        if i in unique:
            idx = np.where(unique == i)[0][0]
            pct = counts[idx] / total * 100
            stats_text += f"{class_name}: {pct:.1f}%\n"
    
    # Show stats in status
    self.show_status(f"✅ Segmentation complete! Ocean: {counts[0]/total*100:.1f}%", 'green')
    
    plt.draw()
    print("Segmentation complete!")
    print(stats_text)

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

def create_progress_bar(ax, title="Progress"):
    """Create a progress bar in given axes"""
    ax.clear()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Progress (%)', fontsize=8)
    ax.set_yticks([])
    
    # Create progress bar rectangle
    progress_rect = Rectangle((0, 0.3), 0, 0.4, 
                            facecolor='green', alpha=0.7)
    ax.add_patch(progress_rect)
    
    return progress_rect

def update_progress_bar(progress_rect, value):
    """Update progress bar to given value (0-100)"""
    progress_rect.set_width(value)
    plt.pause(0.01)

# Example of how to integrate into existing SegmentationTrainer class:
"""
To add these improvements to segmentation_trainer.py:

1. Add status display initialization in __init__:
   self.status_text = None

2. Replace the train_classifier method with enhanced_train_classifier

3. Replace the test_segmentation method with enhanced_test_segmentation  

4. Add the show_status method to the class

5. Add progress updates in compute_features:
   - Show "Computing features..." status
   - Update every few iterations

6. Make test button more visible by adding color:
   self.btn_test = Button(self.ax_test_btn, 'Test', color='lightblue')

7. Add instruction text at startup:
   self.show_status("👆 Click on image to label pixels. Need 100+ samples per class", 'blue')
"""

print("Enhancement functions created. See comments for integration instructions.")