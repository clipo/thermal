# Segmentation Training Improvements

## Overview
Enhanced the segmentation training system with better frame sampling, progress indicators, and area-specific model management.

## Key Improvements

### 1. Better Frame Sampling (`improve_training_sampling.py`)
- **Distributed sampling**: Evenly spaced frames across entire dataset
- **Increment sampling**: Every Nth frame (e.g., every 25th)
- **Random sampling**: Random selection for diversity
- **Multi-directory support**: Handles XXXMEDIA subdirectories
- **Configurable limits**: Max 20 frames by default to prevent overfitting

### 2. Area-Based Model Naming
Models are now named based on the survey area for better organization:
- `24_june_23_segmentation.pkl` - For June 24 survey
- `hanga_roa_rano_kau_segmentation.pkl` - For Hanga Roa area
- `vaihu_west_segmentation.pkl` - For Vaihu West area

### 3. Automatic Model Selection (`sgd_autodetect.py`)
The system now automatically selects the most appropriate model:
1. First checks for area-specific model
2. Falls back to parent area model (for XXXMEDIA subdirs)
3. Uses environment variable if set
4. Falls back to default model

### 4. Enhanced Training Interface (`improve_training_interface.py`)
- **Progress indicators** during training
- **Status messages** with colored borders
- **Real-time feedback** for sample collection
- **Clear requirements**: Shows need for 100+ samples per class
- **Working test visualization**: Fixed blank test panel issue

## Usage Examples

### Training with Better Sampling
```bash
# Train with distributed sampling (recommended)
python sgd_autodetect.py \
    --data "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23" \
    --output june24_sgd.kml \
    --train

# The system will:
# 1. Extract area name: "24_june_23"
# 2. Sample frames evenly (distributed sampling)
# 3. Save model as: models/24_june_23_segmentation.pkl
```

### Processing with Area-Specific Models
```bash
# Process using area-specific model (automatic selection)
python sgd_autodetect.py \
    --data "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23" \
    --output june24_sgd.kml \
    --search

# System automatically uses: models/24_june_23_segmentation.pkl
```

### Multi-Area Processing
```bash
# Process multiple areas with their respective models
python sgd_autodetect.py \
    --data "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights" \
    --output all_flights.kml \
    --search

# Each subdirectory uses its own area-specific model
```

## Files Modified

1. **`sgd_autodetect.py`**
   - Added `select_area_model()` method
   - Updated training to use area-based naming
   - Automatic model selection for each directory

2. **`improve_training_sampling.py`** (NEW)
   - `get_area_name()`: Extract clean area names
   - `find_training_frames()`: Smart frame sampling
   - `create_model_paths()`: Consistent model naming

3. **`improve_training_interface.py`** (NEW)
   - `enhanced_train_classifier()`: Progress during training
   - `enhanced_test_segmentation()`: Fixed test visualization
   - `show_status()`: Visual feedback system

## Benefits

1. **Better Training Data**: Sampling ensures diverse training examples
2. **Organized Models**: Area-based naming prevents confusion
3. **Automatic Selection**: Right model for right area automatically
4. **User Feedback**: Clear progress and status during training
5. **Reduced Errors**: Fixed blank test panel and other UI issues

## Testing

Run the test script to verify area-specific model selection:
```bash
python test_area_model_selection.py
```

This will show:
- Area name extraction from various paths
- Model path generation
- Automatic model selection logic

## Notes

- Models are stored in `models/` directory
- Training data saved alongside models as JSON
- Minimum 100 samples per class recommended
- Maximum 20 frames used for training (configurable)
- Supports processing parent directories with --search flag