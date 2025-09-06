#!/usr/bin/env python3
"""Quick verification of thermal data reading"""

import numpy as np
import struct
from pathlib import Path
from PIL import Image

def verify_thermal():
    """Verify thermal data is being read correctly"""
    frame_num = 248
    base_path = Path("data/100MEDIA")
    
    # Load IRX JPEG
    irx_path = base_path / f"IRX_{frame_num:04d}.jpg"
    irx_image = np.array(Image.open(irx_path))
    print(f"IRX image shape: {irx_image.shape}")
    print(f"IRX dtype: {irx_image.dtype}")
    print(f"IRX range: {irx_image.min()} to {irx_image.max()}")
    
    # Load raw thermal
    irg_path = base_path / f"IRX_{frame_num:04d}.irg"
    with open(irg_path, 'rb') as f:
        irg_data = f.read()
    
    # Extract thermal values
    width, height = 640, 512
    pixel_data_size = width * height * 2
    header_size = len(irg_data) - pixel_data_size
    
    if header_size > 0:
        thermal_raw = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
    else:
        thermal_raw = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
    
    thermal_raw = thermal_raw.reshape((height, width))
    thermal_celsius = (thermal_raw / 10.0) - 273.15
    
    print(f"\nRaw thermal shape: {thermal_celsius.shape}")
    print(f"Temperature range: {np.nanmin(thermal_celsius):.2f}°C to {np.nanmax(thermal_celsius):.2f}°C")
    print(f"Mean temperature: {np.nanmean(thermal_celsius):.2f}°C")
    
    # Sample some ocean and land areas
    ocean_sample = thermal_celsius[200:250, 100:150]  # Ocean area
    land_sample = thermal_celsius[100:150, 450:500]   # Land area
    
    print(f"\nOcean sample mean: {np.nanmean(ocean_sample):.2f}°C")
    print(f"Land sample mean: {np.nanmean(land_sample):.2f}°C")
    print(f"Temperature difference: {abs(np.nanmean(ocean_sample) - np.nanmean(land_sample)):.2f}°C")

if __name__ == "__main__":
    verify_thermal()