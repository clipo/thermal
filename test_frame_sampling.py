#!/usr/bin/env python3
"""
Test different frame sampling methods for segmentation training.
"""

from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

from improve_training_sampling import find_training_frames

def test_sampling_methods():
    """Test all three sampling methods on a test directory."""
    
    # Test directory
    test_path = "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23"
    
    if not Path(test_path).exists():
        print(f"Test path not found: {test_path}")
        print("Using fallback test path...")
        test_path = "."
    
    print("=" * 70)
    print("TESTING FRAME SAMPLING METHODS")
    print("=" * 70)
    print(f"\nTest directory: {test_path}\n")
    
    # Test 1: Distributed sampling (default)
    print("-" * 70)
    print("TEST 1: DISTRIBUTED SAMPLING (evenly spaced)")
    print("-" * 70)
    frames, area = find_training_frames(
        test_path,
        sampling='distributed',
        max_frames=10
    )
    
    if frames:
        frame_numbers = [int(f.stem.split('_')[1]) for f in frames]
        print(f"Selected {len(frames)} frames")
        print(f"Frame numbers: {frame_numbers}")
        if len(frame_numbers) > 1:
            gaps = [frame_numbers[i+1] - frame_numbers[i] for i in range(len(frame_numbers)-1)]
            print(f"Gaps between frames: {gaps}")
            print(f"Average gap: {sum(gaps)/len(gaps):.1f} frames")
    
    # Test 2: Increment sampling
    print("\n" + "-" * 70)
    print("TEST 2: INCREMENT SAMPLING (every Nth frame)")
    print("-" * 70)
    
    for increment in [10, 25, 50]:
        print(f"\nIncrement = {increment}:")
        frames, area = find_training_frames(
            test_path,
            sampling='increment',
            increment=increment,
            max_frames=10
        )
        
        if frames:
            frame_numbers = [int(f.stem.split('_')[1]) for f in frames]
            print(f"  Selected {len(frames)} frames")
            print(f"  Frame numbers: {frame_numbers[:5]}..." if len(frame_numbers) > 5 else f"  Frame numbers: {frame_numbers}")
    
    # Test 3: Random sampling
    print("\n" + "-" * 70)
    print("TEST 3: RANDOM SAMPLING")
    print("-" * 70)
    
    for run in range(3):
        print(f"\nRun {run+1}:")
        frames, area = find_training_frames(
            test_path,
            sampling='random',
            max_frames=10
        )
        
        if frames:
            frame_numbers = sorted([int(f.stem.split('_')[1]) for f in frames])
            print(f"  Selected {len(frames)} frames")
            print(f"  Frame numbers: {frame_numbers[:5]}..." if len(frame_numbers) > 5 else f"  Frame numbers: {frame_numbers}")

def compare_sampling_coverage():
    """Compare coverage of different sampling methods."""
    
    print("\n" + "=" * 70)
    print("SAMPLING COVERAGE COMPARISON")
    print("=" * 70)
    
    test_path = "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23"
    
    if not Path(test_path).exists():
        print("Test path not found, skipping coverage comparison")
        return
    
    # Get all available frames
    all_frames = []
    base_path = Path(test_path)
    for media_dir in sorted(base_path.glob("*MEDIA")):
        all_frames.extend(sorted(media_dir.glob("MAX_*.JPG")))
    
    if not all_frames:
        all_frames = sorted(base_path.glob("MAX_*.JPG"))
    
    total_frames = len(all_frames)
    print(f"\nTotal available frames: {total_frames}")
    
    if total_frames == 0:
        print("No frames found")
        return
    
    # Test different configurations
    configs = [
        ('distributed', None, 20),
        ('increment', 10, 50),
        ('increment', 25, 20),
        ('increment', 50, 10),
        ('random', None, 20),
    ]
    
    print("\n" + "-" * 70)
    print(f"{'Method':<15} {'Increment':<10} {'Max':<5} {'Selected':<10} {'Coverage':<10}")
    print("-" * 70)
    
    for method, increment, max_frames in configs:
        frames, _ = find_training_frames(
            test_path,
            sampling=method,
            increment=increment if increment else 25,
            max_frames=max_frames
        )
        
        selected = len(frames)
        coverage = (selected / total_frames * 100) if total_frames > 0 else 0
        
        inc_str = str(increment) if increment else "N/A"
        print(f"{method:<15} {inc_str:<10} {max_frames:<5} {selected:<10} {coverage:>6.1f}%")

def main():
    print("\n" + "=" * 70)
    print("FRAME SAMPLING TEST SUITE")
    print("=" * 70 + "\n")
    
    # Run tests
    test_sampling_methods()
    compare_sampling_coverage()
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    
    print("""
# Distributed sampling (evenly spaced frames):
python sgd_autodetect.py --data "/path/to/data" --train \\
    --train-sampling distributed --train-max-frames 20

# Increment sampling (every 25th frame):
python sgd_autodetect.py --data "/path/to/data" --train \\
    --train-sampling increment --train-increment 25 --train-max-frames 20

# Random sampling:
python sgd_autodetect.py --data "/path/to/data" --train \\
    --train-sampling random --train-max-frames 15

# Large increment for quick overview:
python sgd_autodetect.py --data "/path/to/data" --train \\
    --train-sampling increment --train-increment 50 --train-max-frames 10
""")
    
    print("=" * 70)

if __name__ == "__main__":
    main()