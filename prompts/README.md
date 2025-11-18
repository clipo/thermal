# SAM Segmentation Prompts

This directory stores SAM (Segment Anything Model) prompt definitions for different environments and conditions.

## What are SAM Prompts?

SAM prompts define class boundaries using point coordinates. Each prompt file specifies:
- **Foreground points**: Locations that belong to a class (label=1)
- **Background points**: Locations that don't belong (label=0)

## Creating Prompts

### Interactive Mode (Recommended)

Launch the interactive prompter to create prompts by clicking on images:

```bash
python scripts/sam_segmenter.py --interactive --data data/100MEDIA
```

**Controls:**
- `1-4`: Select class (ocean/land/rock/wave)
- Left click: Add foreground point
- Right click: Add background point
- `S`: Preview segmentation
- `W`: Save prompts to JSON

### Manual Creation

Copy and modify `example_coastal_rocky.json`:

```json
{
  "ocean": {
    "points": [[x1, y1], [x2, y2], ...],
    "labels": [1, 1, ...]
  },
  "land": {
    "points": [[x3, y3], ...],
    "labels": [1, ...]
  }
}
```

## Using Prompts

### Test on Single Image

```bash
python scripts/compare_segmentation.py \
  --image data/100MEDIA/MAX_0001.JPG \
  --sam-prompts prompts/my_prompts.json
```

### Batch Processing

```bash
python scripts/sam_segmenter.py \
  --data data/100MEDIA \
  --prompts prompts/my_prompts.json \
  --output sgd_output/sam_masks/
```

## Organizing Prompts

Create prompts for different conditions:

```
prompts/
├── rocky_shore.json          # Rocky coastal areas
├── sandy_beach.json          # Sandy beaches
├── sunrise_lighting.json     # Morning/sunset conditions
├── high_waves.json           # Rough ocean conditions
├── calm_ocean.json           # Calm, flat water
└── mixed_terrain.json        # Varied coastal features
```

## Prompt Transferability

- **Same environment**: Prompts transfer perfectly
- **Similar environment**: Often work well with minor adjustments
- **Different environment**: May need new prompts

Test prompts on 3-5 representative frames before batch processing.

## Best Practices

1. **Use 3-5 foreground points per class**
   - Cover different regions (e.g., deep water, shallow water)
   - Include varied lighting conditions

2. **Add background points strategically**
   - At class boundaries (e.g., ocean/land interface)
   - Where misclassification occurs

3. **Create environment-specific prompt libraries**
   - Rocky vs sandy coasts need different prompts
   - Morning vs midday lighting may need separate prompts

4. **Validate before batch processing**
   - Test on 10-20 random frames
   - Check ocean/land boundaries carefully
   - Compare with Random Forest if available

## Troubleshooting

**Poor segmentation results:**
- Add more foreground points in problem areas
- Add background points at misclassified regions
- Try prompts from a different representative frame

**Inconsistent results across frames:**
- Lighting may vary too much - create condition-specific prompts
- Consider using Random Forest for highly variable conditions

**Ocean/land boundary errors:**
- Critical for SGD detection!
- Add extra points at shoreline
- Use background points to exclude rocks from ocean class
