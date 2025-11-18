# SAM Prompts Strategy

## Understanding SAM Prompts

SAM (Segment Anything Model) uses **point prompts** with absolute pixel coordinates (e.g., x=1024, y=768) to guide segmentation. This is fundamentally different from Random Forest, which learns general patterns.

## Key Question: How Many Images Need Prompts?

**Answer: It depends on how much variation exists between your images.**

### Strategy 1: One Prompt Per Flight (RECOMMENDED)

✅ **Use this if images within a flight are consistent:**
- Same altitude and angle
- Similar ocean/land positioning
- Same camera orientation

**How to do it:**
1. Pick one representative image from the flight
2. Create prompts using `sam_prompt_creator.py`
3. Use those prompts for all images in that flight

**Example:**
```bash
# Create prompts for Flight 1
python scripts/sam_prompt_creator.py --image data/flight1/MAX_0050.JPG
# This creates: prompts/sam_MAX_0050_20250118_143022.json

# Use for entire flight
python scripts/sgd_detect_with_sam.py \
  --data data/flight1/ \
  --prompts prompts/sam_MAX_0050_20250118_143022.json
```

### Strategy 2: Multiple Prompts Per Flight

⚠️ **Use this if images vary significantly within a flight:**
- Altitude changes (zoom in/out)
- Direction changes (ocean moves from right to left)
- Lighting/exposure varies dramatically

**How to do it:**
1. Identify 3-5 representative images from different parts of the flight
2. Create prompts for each
3. Match images to appropriate prompts (requires manual work)

### Strategy 3: Prompts for Each Flight Session

📦 **Recommended workflow for multiple flights:**

```bash
# Flight 1 - June 23
python scripts/sam_prompt_creator.py --image data/june23_flight1/MAX_0100.JPG
# Save as: prompts/sam_june23_flight1.json

# Flight 2 - June 24
python scripts/sam_prompt_creator.py --image data/june24_flight2/MAX_0050.JPG
# Save as: prompts/sam_june24_flight2.json

# Process each flight with its prompts
python scripts/sgd_detect_with_sam.py \
  --data data/june23_flight1/ \
  --prompts prompts/sam_june23_flight1.json

python scripts/sgd_detect_with_sam.py \
  --data data/june24_flight2/ \
  --prompts prompts/sam_june24_flight2.json
```

## When SAM Prompts Transfer Well

✅ **Good prompt reuse (same prompts work across images):**
- Consistent flight pattern (e.g., straight line along coast)
- Fixed altitude survey grid
- Similar time of day (consistent shadows)
- Same coastal area

❌ **Poor prompt reuse (need new prompts):**
- Different locations (new coastline = different ocean position)
- Different flight directions (ocean right vs. ocean left)
- Major altitude changes (landscape looks different)
- Different times of day (shadows change everything)

## Testing If Prompts Work

**Quick test:** Apply prompts to 5-10 random images from your flight and visually check:

```bash
# Test prompts on sample images
python scripts/sam_prompt_creator.py --image data/flight1/MAX_0010.JPG
python scripts/sam_prompt_creator.py --image data/flight1/MAX_0050.JPG
python scripts/sam_prompt_creator.py --image data/flight1/MAX_0100.JPG
```

If the segmentation looks good on all test images, **one set of prompts is enough** for that flight.

## Alternative: Random Forest (No Prompts Needed!)

If creating prompts for each flight is too tedious, consider using **Random Forest** instead:

```bash
# Train once on representative images from multiple flights
python scripts/train_ml_segmenter.py \
  --data data/training_samples/ \
  --output models/general_segmentation.pkl

# Use everywhere - no prompts needed!
python scripts/sgd_detect.py \
  --data data/any_flight/ \
  --use-ml \
  --ml-model models/general_segmentation.pkl
```

**Random Forest pros:**
- Train once, use everywhere
- Learns general ocean/land patterns
- No per-flight configuration

**SAM pros:**
- More accurate segmentation
- Better at tricky boundaries (rocky shores, waves)
- No training data collection needed

## Recommendation

**For your Rapa Nui dataset:**

Given that you have multiple flights from different dates and locations:

1. **Start with one flight** - create prompts for one representative image
2. **Test on 5-10 images** from same flight
3. **If it works:** Use those prompts for entire flight
4. **If it doesn't:** You'll need either:
   - Multiple prompt sets per flight, OR
   - Switch to Random Forest for that flight

**Hybrid approach (BEST):**
- Use SAM with prompts for clean, consistent flights
- Use Random Forest for variable/challenging flights
- The wizard supports both!

## Summary

| Scenario | How Many Prompts? |
|----------|-------------------|
| Single consistent flight | **1 set of prompts** |
| Multiple flights, same area | **1 set per flight** |
| Variable altitude/direction | **3-5 sets per flight** |
| Multiple diverse locations | **1 set per location** |
| Too tedious? | **Use Random Forest instead** |

## Creating Prompts

Use the new streamlined tool:

```bash
python scripts/sam_prompt_creator.py --image YOUR_IMAGE.JPG
```

**Interface:**
- Left-click: Add ocean points (blue circles)
- Right-click: Add land points to exclude (red X's)
- Segmentation updates automatically
- Press **W** to save (green border confirms saved)
- Press **Q** to quit

**The window title and border show save status:**
- No border: No prompts yet
- Yellow border: "Press W to Save"
- Green border: "✓ SAVED" with filename
