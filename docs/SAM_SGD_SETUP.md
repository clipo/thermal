# SAM-Based SGD Detection - Setup Guide

Complete guide for reproducing SAM-based SGD detection on any system.

## Quick Setup (5 minutes)

### 1. Install Core Dependencies

```bash
# Clone the repository
git clone https://github.com/clipo/thermal.git
cd thermal

# Install core dependencies
pip install -r requirements.txt
```

### 2. Install SAM (Optional but Recommended)

**Automated Setup (Easiest):**
```bash
bash scripts/setup_sam.sh
```

This script will:
- Install PyTorch (with GPU support if available)
- Install Segment Anything Model
- Download SAM checkpoint weights (you choose size)
- Verify installation

**Manual Setup:**

For NVIDIA GPUs (CUDA):
```bash
# Install PyTorch with CUDA support (check https://pytorch.org for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download checkpoint (choose one):
# ViT-B (375MB, fastest)
wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# ViT-L (1.2GB, better accuracy)
wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# ViT-H (2.5GB, best accuracy)
wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

For Apple Silicon (M1/M2/M3):
```bash
# Install PyTorch with MPS (Metal Performance Shaders) support
pip install torch torchvision

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download checkpoint (ViT-B recommended for M1/M2)
mkdir -p models/sam
wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

For CPU only:
```bash
# Install PyTorch CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Download checkpoint (ViT-B only for CPU)
mkdir -p models/sam
wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. Verify Installation

```bash
# Test SAM installation
python scripts/sam_segmenter.py --test

# Should see:
# ✓ SAM installed correctly
# ✓ PyTorch version: X.X.X
# ✓ Device: cuda/mps/cpu
```

## Using SAM-Based SGD Detection

### Option 1: Via Wizard (Recommended)

```bash
python scripts/sgd_wizard.py
```

1. Choose your data directory
2. Complete segmentation setup (SAM recommended)
3. **In Section 4**, select detection method: **"sam"**
4. Wizard launches interactive SAM SGD detection tool
5. Click on cold spots, press 'S' to segment
6. Compare results with threshold method

### Option 2: Direct Command

```bash
python scripts/test_sam_sgd_detection.py \
    --rgb data/100MEDIA/MAX_0001.JPG \
    --thermal data/100MEDIA/IRX_0001.irg
```

Optional: Use existing ocean mask for better results
```bash
python scripts/test_sam_sgd_detection.py \
    --rgb data/100MEDIA/MAX_0001.JPG \
    --thermal data/100MEDIA/IRX_0001.irg \
    --ocean-mask sgd_output/ocean_masks/MAX_0001_ocean.npy
```

## System Requirements

### Minimum (CPU only)
- Python 3.8+
- 8GB RAM
- ~2GB disk space (with ViT-B model)
- Works but **VERY SLOW** (~60-120 seconds per image)

### Recommended (GPU)
- Python 3.8+
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM, OR
- Apple Silicon M1/M2/M3 with 16GB+ unified memory
- ~2-4GB disk space (with ViT-L model)
- **Fast** (~2-5 seconds per image)

### Optimal (High-end GPU)
- Python 3.8+
- 32GB+ RAM
- NVIDIA GPU with 12GB+ VRAM (RTX 3090, A100, etc.)
- ~4GB disk space (with ViT-H model)
- **Very fast** (~1-2 seconds per image)

## Troubleshooting

### "No module named 'segment_anything'"
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### "Checkpoint not found"
```bash
# Download the checkpoint file
mkdir -p models/sam
wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### SAM is very slow
- Make sure you have GPU support installed (CUDA for NVIDIA, MPS for Apple)
- Check device with: `python -c "import torch; print(torch.cuda.is_available() or torch.backends.mps.is_available())"`
- If False, reinstall PyTorch with GPU support
- Consider using ViT-B instead of ViT-L/ViT-H

### Out of memory errors
- Use smaller model (ViT-B instead of ViT-L/ViT-H)
- Close other applications
- Reduce image resolution if possible

## What You Get

### SAM Ocean/Land Segmentation
- More accurate ocean boundaries
- Better handling of complex coastlines
- Removes landlocked false positives (ponds, lakes)
- GPU-accelerated batch processing

### SAM-Based SGD Detection
- Interactive feature identification
- AI-powered segmentation of cold spots
- Side-by-side comparison with threshold method
- Higher precision for research and validation
- Manual review workflow (not for batch processing)

## When to Use Each Method

| Feature | Threshold | SAM |
|---------|-----------|-----|
| **Speed** | Very fast | Slower (needs GPU) |
| **Setup** | Simple | Requires GPU/model download |
| **Accuracy** | Good | Better |
| **Batch Processing** | ✅ Excellent | ⚠️ Ocean segmentation only |
| **SGD Detection** | ✅ Automated | 👁️ Interactive/manual |
| **Use Case** | Large surveys (100s-1000s images) | Validation, research, complex features |

## Full Reproducibility

For exact environment reproduction:

```bash
# Option 1: Use setup script
bash scripts/setup_sam.sh

# Option 2: Manual installation
pip install -r requirements.txt
# Then install SAM as shown above

# Option 3: Create exact environment
# (if requirements_exact.txt exists)
pip install -r requirements_exact.txt
bash scripts/setup_sam.sh
```

## Next Steps

After setup:
1. Test on sample data: `python scripts/test_sam_sgd_detection.py --rgb <RGB> --thermal <THERMAL>`
2. Run wizard: `python scripts/sgd_wizard.py`
3. Read full guide: `scripts/SAM_SGD_DETECTION_GUIDE.md`
4. Process your survey data

## Questions?

- Technical Paper: `TECHNICAL_PAPER.md`
- SAM Quick Start: `docs/SAM_QUICKSTART.md`
- Main README: `README.md`
- Open an issue: https://github.com/clipo/thermal/issues
