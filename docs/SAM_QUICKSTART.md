# SAM Quick Start Guide

Get started with SAM (Segment Anything Model) for improved ocean/land segmentation in 5 minutes.

## Prerequisites

- NVIDIA GPU with CUDA support (required)
- Python environment with sgd-toolkit installed
- Sample thermal imagery in `data/` directory

## Step 1: Install SAM (2 minutes)

```bash
cd /Users/clipo/PycharmProjects/thermal
bash scripts/setup_sam.sh
```

**Choose model size:**
- Type `1` for ViT-H (Huge) - Best accuracy, recommended for DGX
- Type `2` for ViT-L (Large) - Good accuracy, faster
- Type `3` for ViT-B (Base) - Fastest, good for testing

**Wait for download** (2.5GB for ViT-H, 1.2GB for ViT-L, 375MB for ViT-B)

## Step 2: Test Installation (30 seconds)

```bash
python scripts/sam_segmenter.py --test
```

**Expected output:**
```
✓ SAM is installed and working!
✓ Using device: CUDA
✓ GPU: NVIDIA [Your GPU Model]
```

**If you see errors**, check:
- GPU drivers installed: `nvidia-smi`
- CUDA PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## Step 3: Create Prompts (2 minutes)

Launch the interactive prompter on a representative image:

```bash
python scripts/sam_segmenter.py --interactive --data data/100MEDIA
```

**A window will open with three panels:**

### Quick Workflow:

1. **Press `1`** to select Ocean class
2. **Left-click 3-5 locations** in ocean (deep water, shallow water)
3. **Press `2`** to select Land class
4. **Left-click 3-5 locations** on land (vegetation, sand, rocks)
5. **Press `3`** to select Rock class (optional)
6. **Left-click 3-5 locations** on rocky shores
7. **Press `S`** to preview segmentation
8. **Press `W`** to save prompts
9. **Press `Q`** to quit

**Prompts saved to:** `prompts/sam_prompts_TIMESTAMP.json`

### Pro Tips:
- Right-click for background points (exclude from class)
- Press `S` frequently to preview results
- Press `C` to clear current class and start over
- Use diverse examples (different lighting, depths, etc.)

## Step 4: Compare with Random Forest (1 minute)

Test SAM vs your existing Random Forest model:

```bash
python scripts/compare_segmentation.py \
  --image data/100MEDIA/MAX_0001.JPG \
  --rf-model models/segmentation_model.pkl \
  --sam-prompts prompts/sam_prompts_20250118_*.json
```

**This shows:**
- Side-by-side comparison
- Class distribution statistics
- Difference analysis

**Look for:**
- More accurate ocean/land boundaries
- Better handling of rocky shores
- Fewer false positives at land edges

## Step 5: Batch Process (Optional)

If SAM performs better, process your entire survey:

```bash
python scripts/sam_segmenter.py \
  --data data/100MEDIA \
  --prompts prompts/sam_prompts_20250118_*.json \
  --output sgd_output/sam_masks/
```

## Next Steps

### Create Environment-Specific Prompts

Different conditions may need different prompts:

```bash
# Rocky shore
python scripts/sam_segmenter.py --interactive --data data/rocky_shore/
# Save as: prompts/rocky_shore.json

# Sandy beach
python scripts/sam_segmenter.py --interactive --data data/sandy_beach/
# Save as: prompts/sandy_beach.json

# Sunrise/sunset
python scripts/sam_segmenter.py --interactive --data data/morning_flight/
# Save as: prompts/sunrise_lighting.json
```

### Integrate with SGD Detection Pipeline

See main README section "SAM Integration with SGD Detection Pipeline" for details on replacing FastMLSegmenter with SAM in your detection workflow.

## Troubleshooting

### "CUDA out of memory"

**Solution 1:** Use smaller model
```bash
python scripts/sam_segmenter.py --model vit_b --interactive
```

**Solution 2:** Close other GPU applications
```bash
nvidia-smi  # Check GPU usage
# Kill other processes using GPU
```

### "SAM not installed"

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Poor segmentation results

**Add more foreground points:**
- Ocean: Include deep water, shallow water, different lighting
- Land: Include vegetation, sand, dry rocks
- Rock: Include wet rocks, dry rocks, various colors

**Add background points:**
- Right-click at ocean/land boundary
- Right-click where misclassification occurs

**Try different frame:**
- Some frames are more representative than others
- Use frames with clear class boundaries

### Slow processing

**Check GPU is being used:**
```bash
python scripts/sam_segmenter.py --test
# Should show: "Using device: cuda"
```

**If showing "cpu":**
```bash
# Install CUDA PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Performance Comparison

| Method | Hardware | Speed | Accuracy | Training Required |
|--------|----------|-------|----------|-------------------|
| **Random Forest** | CPU | 0.08-0.15s/frame | 88-99% | Yes (per environment) |
| **SAM (ViT-H)** | GPU | 0.3-0.5s/frame | 92-99% | No (prompts only) |
| **SAM (ViT-L)** | GPU | 0.2-0.3s/frame | 90-98% | No (prompts only) |
| **SAM (ViT-B)** | GPU | 0.1-0.2s/frame | 88-96% | No (prompts only) |

## When to Use What

### Use SAM when:
- You have GPU available (NVIDIA with CUDA)
- Working with new/varied environments
- Need best boundary accuracy
- Want to avoid retraining for each survey

### Use Random Forest when:
- No GPU available
- Need fastest CPU processing
- Working with consistent environment
- Already have well-trained model

### Hybrid Approach:
- Use SAM for high-value surveys (DGX processing)
- Use Random Forest for field work (laptop)
- Both produce compatible masks

## Support

For detailed documentation, see:
- Main README: Section "SAM (Segment Anything Model)"
- Prompts guide: `prompts/README.md`
- Model details: `models/sam/README.md`

For issues:
- Check troubleshooting section above
- Review SAM documentation: https://github.com/facebookresearch/segment-anything
- Check CUDA installation: `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`
