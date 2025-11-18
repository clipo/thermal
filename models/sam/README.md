# SAM Model Checkpoints

This directory stores SAM (Segment Anything Model) checkpoint files.

## Installation

Run the setup script to download checkpoints:

```bash
bash scripts/setup_sam.sh
```

This will prompt you to choose a model size:

### Model Options

| Model | Size | Accuracy | Speed | Recommended For |
|-------|------|----------|-------|-----------------|
| **ViT-H** | 2.5GB | Best | Slower | Workstations, DGX systems |
| **ViT-L** | 1.2GB | Good | Medium | General purpose |
| **ViT-B** | 375MB | Good | Fastest | Testing, laptops |

### Expected Files

After installation, you should see:

```
models/sam/
├── sam_vit_h_4b8939.pth  (if you chose ViT-H)
├── sam_vit_l_0b3195.pth  (if you chose ViT-L)
└── sam_vit_b_01ec64.pth  (if you chose ViT-B)
```

## Manual Download

If automatic download fails, download manually:

1. **ViT-H (Huge) - 2.5GB**
   ```bash
   wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   ```

2. **ViT-L (Large) - 1.2GB**
   ```bash
   wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
   ```

3. **ViT-B (Base) - 375MB**
   ```bash
   wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```

## Using Different Models

Specify model type when running SAM scripts:

```bash
# Use ViT-H (default)
python scripts/sam_segmenter.py --model vit_h --interactive

# Use ViT-L (faster)
python scripts/sam_segmenter.py --model vit_l --interactive

# Use ViT-B (fastest)
python scripts/sam_segmenter.py --model vit_b --interactive
```

## GPU Requirements

| Model | VRAM Required |
|-------|---------------|
| ViT-H | ~8GB |
| ViT-L | ~4GB |
| ViT-B | ~2GB |

## Testing Installation

```bash
python scripts/sam_segmenter.py --test
```

Should output:
```
✓ SAM is installed and working!
✓ Using device: CUDA
✓ GPU: [Your GPU Name]
```

## Troubleshooting

**"Checkpoint not found" error:**
- Run `bash scripts/setup_sam.sh` to download
- Verify .pth file exists in this directory
- Check file isn't corrupted (should be hundreds of MB)

**"CUDA out of memory" error:**
- Use smaller model (ViT-B or ViT-L instead of ViT-H)
- Close other GPU applications
- Check available VRAM with `nvidia-smi`

**No GPU detected:**
- Install CUDA toolkit
- Install PyTorch with CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
