#!/usr/bin/env python3
"""
Improvements to segmentation training for better frame sampling and model naming.
"""

import numpy as np
from pathlib import Path
import random

def get_area_name(data_path):
    """
    Get the area name from the directory structure.
    E.g., "/Volumes/RapaNui/Rapa Nui Jan 2024/Autel/Flight 3 - Vaihu - West" -> "vaihu_west"
    """
    data_path = Path(data_path)
    
    # If it's a XXXMEDIA directory, go up one level
    if data_path.name.endswith('MEDIA'):
        area_path = data_path.parent
    else:
        area_path = data_path
    
    # Clean the area name for file naming
    area_name = area_path.name.lower()
    
    # Remove common prefixes
    area_name = area_name.replace('flight ', '').replace('flight_', '')
    area_name = area_name.replace(' - ', '_').replace('-', '_')
    area_name = area_name.replace(' ', '_')
    
    # Remove non-alphanumeric characters except underscore
    area_name = ''.join(c if c.isalnum() or c == '_' else '' for c in area_name)
    
    # Ensure it's not empty
    if not area_name:
        area_name = 'unnamed_area'
    
    return area_name

def find_training_frames(base_path, sampling='increment', increment=25, max_frames=20):
    """
    Find frames for training with better sampling strategy.
    
    Args:
        base_path: Directory containing images or XXXMEDIA subdirectories
        sampling: 'increment' (every Nth frame), 'random', or 'distributed'
        increment: If using increment sampling, skip this many frames
        max_frames: Maximum number of frames to use for training
    
    Returns:
        List of frame paths and area name
    """
    base_path = Path(base_path)
    frames = []
    
    # Check if we're in a directory with XXXMEDIA subdirectories
    media_dirs = sorted([d for d in base_path.iterdir() 
                         if d.is_dir() and d.name.endswith('MEDIA')])
    
    if media_dirs:
        print(f"Found {len(media_dirs)} MEDIA directories")
        # Collect frames from all MEDIA directories
        for media_dir in media_dirs:
            dir_frames = sorted(media_dir.glob("MAX_*.JPG"))
            if dir_frames:
                frames.extend(dir_frames)
                print(f"  {media_dir.name}: {len(dir_frames)} frames")
    else:
        # Single directory
        frames = sorted(base_path.glob("MAX_*.JPG"))
        print(f"Found {len(frames)} frames in {base_path.name}")
    
    if not frames:
        print("No frames found!")
        return [], None
    
    # Get area name for model naming
    area_name = get_area_name(base_path)
    print(f"Area name for model: {area_name}")
    
    # Sample frames based on strategy
    if sampling == 'increment':
        # Take every Nth frame
        sampled_frames = frames[::increment]
        print(f"Increment sampling: every {increment}th frame = {len(sampled_frames)} frames")
        
    elif sampling == 'random':
        # Random sampling
        num_samples = min(max_frames, len(frames))
        sampled_frames = random.sample(frames, num_samples)
        print(f"Random sampling: {num_samples} frames from {len(frames)} total")
        
    elif sampling == 'distributed':
        # Evenly distributed across all frames
        num_samples = min(max_frames, len(frames))
        if num_samples == len(frames):
            sampled_frames = frames
        else:
            # Calculate indices for even distribution
            indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)
            sampled_frames = [frames[i] for i in indices]
        print(f"Distributed sampling: {len(sampled_frames)} frames evenly spaced")
    
    else:
        # Default to all frames up to max
        sampled_frames = frames[:max_frames]
    
    # Limit to max_frames
    if len(sampled_frames) > max_frames:
        sampled_frames = sampled_frames[:max_frames]
        print(f"Limited to {max_frames} frames maximum")
    
    return sampled_frames, area_name

def create_model_paths(area_name):
    """
    Create model and training data paths based on area name.
    
    Returns:
        model_path, training_path
    """
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / f"{area_name}_segmentation.pkl"
    training_path = models_dir / f"{area_name}_training.json"
    
    return model_path, training_path

def enhance_training_initialization(base_path, sampling='distributed'):
    """
    Enhanced initialization for SegmentationTrainer with better sampling.
    """
    # Find frames with better sampling
    frames, area_name = find_training_frames(
        base_path, 
        sampling=sampling,
        increment=25,
        max_frames=20
    )
    
    if not frames:
        raise ValueError(f"No frames found in {base_path}")
    
    # Create model paths based on area
    model_path, training_path = create_model_paths(area_name)
    
    print(f"\nTraining Configuration:")
    print(f"  Area: {area_name}")
    print(f"  Frames: {len(frames)}")
    print(f"  Model: {model_path}")
    print(f"  Training data: {training_path}")
    print(f"  Sampling: {sampling}")
    
    # Extract frame numbers for display
    frame_numbers = []
    for frame_path in frames:
        # Extract frame number from filename (e.g., MAX_0123.JPG -> 123)
        try:
            num = int(frame_path.stem.split('_')[1])
            frame_numbers.append(num)
        except:
            pass
    
    if frame_numbers:
        print(f"  Frame numbers: {frame_numbers[:5]}..." if len(frame_numbers) > 5 else f"  Frame numbers: {frame_numbers}")
    
    return {
        'frames': frames,
        'area_name': area_name,
        'model_path': model_path,
        'training_path': training_path,
        'frame_numbers': frame_numbers
    }

def update_sgd_autodetect_model_selection(data_path):
    """
    Logic to select the appropriate model based on the survey area.
    """
    area_name = get_area_name(data_path)
    
    # Check for area-specific model
    area_model = Path('models') / f"{area_name}_segmentation.pkl"
    
    if area_model.exists():
        print(f"Using area-specific model: {area_model}")
        return area_model
    
    # Check for generic model in same directory structure
    parent_name = get_area_name(data_path.parent) if data_path.parent else None
    if parent_name:
        parent_model = Path('models') / f"{parent_name}_segmentation.pkl"
        if parent_model.exists():
            print(f"Using parent area model: {parent_model}")
            return parent_model
    
    # Fall back to default model
    default_model = Path('segmentation_model.pkl')
    if default_model.exists():
        print(f"Using default model: {default_model}")
        return default_model
    
    print("No suitable segmentation model found")
    return None

# Example usage in segmentation_trainer.py:
"""
To integrate into segmentation_trainer.py:

1. Replace the __init__ frame finding with:

    # Import the enhanced functions
    from improve_training_sampling import find_training_frames, create_model_paths
    
    # In __init__:
    self.frames, self.area_name = find_training_frames(
        base_path, 
        sampling='distributed',  # or 'increment' or 'random'
        increment=25,
        max_frames=20
    )
    
    # Create model paths based on area
    self.model_file, self.training_file = create_model_paths(self.area_name)
    
    print(f"Training for area: {self.area_name}")
    print(f"Using {len(self.frames)} frames")

2. Update the frame loading to work with full paths:

    def load_frame(self, frame_path):
        # Handle both frame number and path
        if isinstance(frame_path, (int, str)) and not Path(frame_path).exists():
            # Old style - frame number
            frame_path = self.base_path / f"MAX_{frame_path:04d}.JPG"
        
        # Load the actual image
        self.rgb_image = Image.open(frame_path)
        ...

3. In sgd_autodetect.py, update model selection:

    from improve_training_sampling import update_sgd_autodetect_model_selection
    
    # When initializing segmenter:
    model_path = update_sgd_autodetect_model_selection(args.data)
    if model_path:
        segmenter = MLSegmenter(model_path=model_path)
"""

if __name__ == "__main__":
    # Test the functions
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "/Volumes/RapaNui/Rapa Nui Jan 2024/Autel/Flight 3 - Vaihu - West"
    
    print(f"Testing with: {test_path}")
    
    # Test area name extraction
    area = get_area_name(test_path)
    print(f"Area name: {area}")
    
    # Test frame finding
    frames, area_name = find_training_frames(test_path, sampling='distributed')
    
    # Test model paths
    if area_name:
        model_path, training_path = create_model_paths(area_name)
        print(f"Model would be saved to: {model_path}")
        print(f"Training data would be saved to: {training_path}")