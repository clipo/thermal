#!/usr/bin/env python3
"""
Test script to verify area-specific model selection is working correctly.
"""

from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

from improve_training_sampling import get_area_name, create_model_paths

def test_area_naming():
    """Test area name extraction from various paths."""
    
    test_paths = [
        "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23",
        "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/102MEDIA",
        "/Volumes/RapaNui/Rapa Nui Jan 2024/Autel/Flight 3 - Vaihu - West",
        "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/1 July 23/Hanga Roa - Rano Kau",
    ]
    
    print("Testing area name extraction:")
    print("-" * 60)
    
    for path in test_paths:
        area_name = get_area_name(path)
        model_path, training_path = create_model_paths(area_name)
        
        print(f"\nPath: {path}")
        print(f"  Area name: {area_name}")
        print(f"  Model: {model_path}")
        print(f"  Training: {training_path}")

def test_model_selection():
    """Test the model selection logic."""
    
    print("\n\nTesting model selection logic:")
    print("-" * 60)
    
    # Import the SGDAutoDetector to test its select_area_model method
    from sgd_autodetect import SGDAutoDetector
    
    # Create a mock detector just to test the method
    class MockDetector:
        def __init__(self):
            self.verbose = True
        
        def select_area_model(self, data_path):
            """Copy of the select_area_model method"""
            from improve_training_sampling import get_area_name
            import os
            
            # Get area name from directory structure
            area_name = get_area_name(data_path)
            
            # Check for area-specific model
            models_dir = Path('models')
            area_model = models_dir / f"{area_name}_segmentation.pkl"
            
            if area_model.exists():
                if self.verbose:
                    print(f"✓ Using area-specific model: {area_model.name}")
                return str(area_model)
            
            # Check for parent area model if in a XXXMEDIA subdirectory
            if Path(data_path).name.endswith('MEDIA'):
                parent_name = get_area_name(Path(data_path).parent)
                parent_model = models_dir / f"{parent_name}_segmentation.pkl"
                if parent_model.exists():
                    if self.verbose:
                        print(f"✓ Using parent area model: {parent_model.name}")
                    return str(parent_model)
            
            # Check environment variable
            env_model = os.environ.get('SGD_MODEL_PATH')
            if env_model and Path(env_model).exists():
                if self.verbose:
                    print(f"✓ Using model from environment: {env_model}")
                return env_model
            
            # Fall back to default model
            default_model = Path('segmentation_model.pkl')
            if default_model.exists():
                if self.verbose:
                    print(f"✓ Using default model: {default_model.name}")
                return str(default_model)
            
            # No model found - will need to train
            if self.verbose:
                print("⚠ No segmentation model found - training will be required")
            return 'segmentation_model.pkl'
    
    detector = MockDetector()
    
    # Test with sample paths
    test_cases = [
        "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23",
        "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/102MEDIA",
    ]
    
    for path in test_cases:
        print(f"\nTesting path: {path}")
        area_name = get_area_name(path)
        print(f"  Area name: {area_name}")
        
        selected_model = detector.select_area_model(path)
        print(f"  Selected model: {selected_model}")

def main():
    print("\n" + "=" * 60)
    print("AREA-SPECIFIC MODEL SELECTION TEST")
    print("=" * 60 + "\n")
    
    test_area_naming()
    test_model_selection()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    # Show existing models
    models_dir = Path('models')
    if models_dir.exists():
        print("\nExisting models in models/ directory:")
        pkl_files = list(models_dir.glob("*.pkl"))
        if pkl_files:
            for model in sorted(pkl_files):
                print(f"  - {model.name}")
        else:
            print("  (no models found)")
    else:
        print("\nModels directory not found - will be created when training")

if __name__ == "__main__":
    main()