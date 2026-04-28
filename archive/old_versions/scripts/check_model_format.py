#!/usr/bin/env python3
"""Check the format of the segmentation model."""

import pickle
from pathlib import Path

model_path = Path('/Users/clipo/PycharmProjects/thermal/segmentation_model.pkl')

with open(model_path, 'rb') as f:
    data = pickle.load(f)
    
print("Model data type:", type(data))
print("Model data keys/attributes:")

if isinstance(data, dict):
    for key in data.keys():
        print(f"  - {key}: {type(data[key])}")
        if key == 'model':
            print(f"    Model type: {type(data[key])}")
            if hasattr(data[key], 'n_features_in_'):
                print(f"    Features expected: {data[key].n_features_in_}")
else:
    print("  Model object:", data)
    if hasattr(data, 'n_features_in_'):
        print(f"  Features expected: {data.n_features_in_}")