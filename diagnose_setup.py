#!/usr/bin/env python3
"""
Diagnostic script to check SGD detection setup consistency
Helps identify why results might differ across computers
"""

import sys
import os
import hashlib
import json
import numpy as np
from pathlib import Path

def get_file_hash(filepath):
    """Get MD5 hash of a file"""
    if not os.path.exists(filepath):
        return "FILE NOT FOUND"
    
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def check_versions():
    """Check versions of critical packages"""
    print("="*60)
    print("PACKAGE VERSIONS")
    print("="*60)
    
    packages = {
        'numpy': None,
        'matplotlib': None,
        'PIL': None,
        'scipy': None,
        'skimage': None,
        'sklearn': None,
    }
    
    for pkg_name in packages:
        try:
            if pkg_name == 'PIL':
                import PIL
                packages[pkg_name] = PIL.__version__
            elif pkg_name == 'skimage':
                import skimage
                packages[pkg_name] = skimage.__version__
            elif pkg_name == 'sklearn':
                import sklearn
                packages[pkg_name] = sklearn.__version__
            else:
                pkg = __import__(pkg_name)
                packages[pkg_name] = pkg.__version__
        except ImportError:
            packages[pkg_name] = "NOT INSTALLED"
        except AttributeError:
            packages[pkg_name] = "Version unknown"
    
    for pkg, version in packages.items():
        status = "✓" if version and version != "NOT INSTALLED" else "✗"
        print(f"{status} {pkg:15} {version}")
    
    print(f"\nPython version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    return packages

def check_model_files():
    """Check integrity of model files"""
    print("\n" + "="*60)
    print("MODEL FILES")
    print("="*60)
    
    files_to_check = {
        'segmentation_model.pkl': '(ML model)',
        'segmentation_training_data.json': '(Training data)',
    }
    
    hashes = {}
    for filename, desc in files_to_check.items():
        filepath = Path(filename)
        if filepath.exists():
            file_hash = get_file_hash(filepath)
            file_size = filepath.stat().st_size
            print(f"✓ {filename:35} {desc}")
            print(f"  Size: {file_size:,} bytes")
            print(f"  MD5:  {file_hash}")
            hashes[filename] = {'size': file_size, 'md5': file_hash}
        else:
            print(f"✗ {filename:35} NOT FOUND")
            hashes[filename] = {'size': 0, 'md5': 'NOT FOUND'}
    
    return hashes

def check_random_seed():
    """Check if random seed affects results"""
    print("\n" + "="*60)
    print("RANDOM SEED TEST")
    print("="*60)
    
    # Test numpy random
    np.random.seed(42)
    test_values = np.random.rand(5)
    expected = np.array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864])
    
    if np.allclose(test_values, expected):
        print("✓ NumPy random number generation is consistent")
    else:
        print("✗ NumPy random number generation differs!")
        print(f"  Expected: {expected}")
        print(f"  Got:      {test_values}")
    
    # Test sklearn random
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Create dummy data
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        
        # Train with fixed seed
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        # Test prediction
        test_X = np.array([[0.5, 0.5, 0.5, 0.5]])
        pred = clf.predict(test_X)[0]
        
        print(f"✓ Scikit-learn random forest seed test: prediction = {pred}")
        
    except Exception as e:
        print(f"✗ Could not test sklearn: {e}")

def check_image_loading():
    """Check image loading consistency"""
    print("\n" + "="*60)
    print("IMAGE LOADING TEST")
    print("="*60)
    
    # Find a test image
    test_patterns = [
        "data/100MEDIA/MAX_0001.JPG",
        "data/100MEDIA/IRX_0001.irg",
    ]
    
    for pattern in test_patterns:
        if Path(pattern).exists():
            print(f"\nTesting: {pattern}")
            
            if pattern.endswith('.JPG'):
                # Test RGB image
                try:
                    from PIL import Image
                    img = Image.open(pattern)
                    img_array = np.array(img)
                    print(f"  Shape: {img_array.shape}")
                    print(f"  Dtype: {img_array.dtype}")
                    print(f"  Mean pixel value: {img_array.mean():.2f}")
                    print(f"  Std pixel value: {img_array.std():.2f}")
                    
                    # Check EXIF
                    exif = img._getexif()
                    if exif:
                        print(f"  EXIF tags found: {len(exif)}")
                    else:
                        print("  WARNING: No EXIF data found!")
                        
                except Exception as e:
                    print(f"  ERROR: {e}")
                    
            elif pattern.endswith('.irg'):
                # Test thermal image
                try:
                    with open(pattern, 'rb') as f:
                        data = f.read()
                    print(f"  File size: {len(data)} bytes")
                    
                    # Try to parse thermal data
                    thermal_width = 640
                    thermal_height = 512
                    expected_size = thermal_width * thermal_height * 2
                    
                    if len(data) >= expected_size:
                        header_size = len(data) - expected_size
                        raw_thermal = np.frombuffer(
                            data[header_size:] if header_size > 0 else data[:expected_size],
                            dtype=np.uint16
                        ).reshape((thermal_height, thermal_width))
                        
                        temp_celsius = (raw_thermal / 10.0) - 273.15
                        print(f"  Temperature range: {temp_celsius.min():.1f}°C to {temp_celsius.max():.1f}°C")
                        print(f"  Mean temperature: {temp_celsius.mean():.1f}°C")
                    else:
                        print(f"  WARNING: File size mismatch. Expected {expected_size}, got {len(data)}")
                        
                except Exception as e:
                    print(f"  ERROR: {e}")
        else:
            print(f"✗ {pattern} not found")

def check_ml_model():
    """Test ML model predictions"""
    print("\n" + "="*60)
    print("ML MODEL CONSISTENCY TEST")
    print("="*60)
    
    try:
        import pickle
        
        # Load model
        with open('segmentation_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model type: {type(model).__name__}")
        
        if hasattr(model, 'n_estimators'):
            print(f"Number of trees: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"Max depth: {model.max_depth}")
        if hasattr(model, 'random_state'):
            print(f"Random state: {model.random_state}")
        
        # Create test data (same seed for consistency)
        np.random.seed(42)
        test_features = np.random.rand(10, 48)  # 48 features as expected
        
        # Predict
        predictions = model.predict(test_features)
        print(f"\nTest predictions: {predictions}")
        print(f"Unique classes: {np.unique(predictions)}")
        
        # Check feature importance
        if hasattr(model, 'feature_importances_'):
            top_features = np.argsort(model.feature_importances_)[-5:]
            print(f"Top 5 important features: {top_features}")
            
    except FileNotFoundError:
        print("✗ segmentation_model.pkl not found")
    except Exception as e:
        print(f"✗ Error testing model: {e}")

def suggest_fixes():
    """Suggest potential fixes"""
    print("\n" + "="*60)
    print("POTENTIAL ISSUES AND FIXES")
    print("="*60)
    
    print("""
Common causes of different results across computers:

1. **Different package versions**
   Fix: pip install -r requirements.txt
   
2. **Missing or corrupted model file**
   Fix: git pull to get latest model
   
3. **Different random seeds**
   Fix: Ensure random_state is set in ML model
   
4. **Platform differences (Windows/Mac/Linux)**
   Fix: Use consistent file paths and encoding
   
5. **Different Python versions**
   Fix: Use Python 3.8+ consistently
   
6. **GPU vs CPU processing**
   Fix: Force CPU-only mode for consistency
   
7. **Image color profile differences**
   Fix: Ensure PIL/Pillow versions match

To ensure reproducibility:
1. Clone fresh repository: git clone https://github.com/clipo/thermal.git
2. Create new virtual environment
3. Install exact package versions: pip install -r requirements.txt
4. Verify model files match (check MD5 hashes above)
""")

def main():
    print("SGD DETECTION SETUP DIAGNOSTIC")
    print("=" * 60)
    
    # Run all checks
    versions = check_versions()
    model_hashes = check_model_files()
    check_random_seed()
    check_image_loading()
    check_ml_model()
    suggest_fixes()
    
    # Save diagnostic report
    report = {
        'timestamp': str(datetime.now()),
        'platform': sys.platform,
        'python_version': sys.version,
        'package_versions': versions,
        'model_files': model_hashes
    }
    
    with open('diagnostic_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("Diagnostic report saved to: diagnostic_report.json")
    print("Share this file when reporting reproducibility issues")
    print("="*60)

if __name__ == "__main__":
    from datetime import datetime
    main()