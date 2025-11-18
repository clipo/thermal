#!/usr/bin/env python3
"""
SAM-based Ocean/Land Segmentation for SGD Detection Pipeline

Wrapper around Meta's Segment Anything Model (SAM) that provides the same
interface as FastMLSegmenter for drop-in replacement.
"""

import numpy as np
from pathlib import Path
import json

try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class SAMSegmenter:
    """
    SAM-based segmenter compatible with SGD detection pipeline.

    Provides the same interface as FastMLSegmenter but uses SAM with
    pre-defined prompts instead of Random Forest classification.
    """

    def __init__(self, model_type='auto', checkpoint_path=None, prompts_path=None):
        """
        Initialize SAM segmenter

        Args:
            model_type: 'vit_h', 'vit_l', 'vit_b', or 'auto' (default: auto)
            checkpoint_path: Path to SAM checkpoint (auto-detected if None)
            prompts_path: Path to prompts JSON file
        """
        if not SAM_AVAILABLE:
            raise ImportError("SAM not installed. Run: bash scripts/setup_sam.sh")

        # Auto-select model based on available hardware
        if model_type == 'auto':
            model_type = self._auto_select_model()

        # Auto-detect checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._find_checkpoint(model_type)

        # Load SAM model
        print(f"Loading SAM model: {model_type}")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        # Use best available device
        device = self._get_device()
        sam.to(device=device)
        print(f"SAM using device: {device}")

        self.predictor = SamPredictor(sam)
        self.device = device
        self.classes = ['ocean', 'land', 'rock', 'wave']

        # Load prompts if provided
        self.prompts = None
        if prompts_path and Path(prompts_path).exists():
            self.load_prompts(prompts_path)

    def _get_device(self):
        """Get best available device (CUDA > MPS > CPU)"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except:
            pass
        return "cpu"

    def _auto_select_model(self):
        """
        Auto-select best SAM model based on available hardware

        Returns:
            'vit_h' for powerful NVIDIA GPUs (DGX, A100, etc.)
            'vit_l' for mid-range NVIDIA GPUs
            'vit_b' for Apple Silicon (M1/M2/M3) or CPU
        """
        try:
            import torch

            if torch.cuda.is_available():
                # NVIDIA GPU available
                device_name = torch.cuda.get_device_name(0).lower()
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

                print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {total_memory:.1f} GB")

                # DGX, A100, H100, or high-end GPUs
                if any(x in device_name for x in ['a100', 'h100', 'v100', 'dgx']):
                    print("→ Using ViT-H (best accuracy for high-end GPU)")
                    return 'vit_h'

                # Mid to high-end consumer/workstation GPUs
                elif total_memory >= 10:  # 10GB+ VRAM
                    print("→ Using ViT-L (good balance for powerful GPU)")
                    return 'vit_l'

                # Lower-end GPUs
                else:
                    print("→ Using ViT-B (fastest for GPU)")
                    return 'vit_b'

            elif torch.backends.mps.is_available():
                # Apple Silicon (M1/M2/M3)
                print("Detected: Apple Silicon (M1/M2/M3)")
                print("→ Using ViT-B (optimized for Apple Silicon)")
                return 'vit_b'

            else:
                # CPU only
                print("No GPU detected, using CPU")
                print("→ Using ViT-B (smallest model for CPU)")
                print("⚠️  Warning: SAM on CPU will be very slow!")
                return 'vit_b'

        except Exception as e:
            print(f"Warning: Could not detect hardware: {e}")
            print("→ Defaulting to ViT-B")
            return 'vit_b'

    def _find_checkpoint(self, model_type):
        """Auto-detect SAM checkpoint file"""
        models_dir = Path("models/sam")

        checkpoint_names = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_b': 'sam_vit_b_01ec64.pth'
        }

        checkpoint = models_dir / checkpoint_names[model_type]

        if not checkpoint.exists():
            # Try to find any .pth file
            pth_files = list(models_dir.glob("*.pth"))
            if pth_files:
                print(f"Using checkpoint: {pth_files[0]}")
                return str(pth_files[0])

            raise FileNotFoundError(
                f"SAM checkpoint not found at {checkpoint}\n"
                f"Run: bash scripts/setup_sam.sh"
            )

        return str(checkpoint)

    def load_prompts(self, prompts_path):
        """Load prompts from JSON file"""
        with open(prompts_path, 'r') as f:
            self.prompts = json.load(f)
        print(f"Loaded SAM prompts: {prompts_path}")

    def segment_ultra_fast(self, rgb_image):
        """
        Segment image using SAM (compatible with FastMLSegmenter interface)

        Args:
            rgb_image: RGB image (H x W x 3)

        Returns:
            dict with 'ocean', 'land', 'waves' masks (same as FastMLSegmenter)
        """
        if self.prompts is None:
            raise ValueError("No prompts loaded. Call load_prompts() first.")

        # Ensure uint8
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)

        # Set image for SAM
        self.predictor.set_image(rgb_image)

        h, w = rgb_image.shape[:2]

        # Initialize masks
        ocean_mask = np.zeros((h, w), dtype=bool)
        land_mask = np.zeros((h, w), dtype=bool)
        rock_mask = np.zeros((h, w), dtype=bool)
        wave_mask = np.zeros((h, w), dtype=bool)

        # Segment each class using prompts
        for class_name, prompt_data in self.prompts.items():
            if class_name not in self.classes:
                continue

            if not prompt_data.get('points'):
                continue

            # Prepare prompts
            points = np.array(prompt_data['points'])
            labels = np.array(prompt_data['labels'])

            # Segment
            masks, scores, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )

            # Use best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            # Assign to appropriate class
            if class_name == 'ocean':
                ocean_mask |= mask
            elif class_name == 'land':
                land_mask |= mask
            elif class_name == 'rock':
                rock_mask |= mask
            elif class_name == 'wave':
                wave_mask |= mask

        # Clean up masks (similar to FastMLSegmenter)
        from skimage import morphology, measure

        # Remove small objects
        if np.any(ocean_mask):
            ocean_mask = morphology.remove_small_objects(ocean_mask, min_size=100)

        if np.any(land_mask):
            land_mask = morphology.remove_small_objects(land_mask, min_size=100)

        # Keep only largest ocean contiguous area
        if np.any(ocean_mask):
            ocean_labels = measure.label(ocean_mask, connectivity=2)
            if ocean_labels.max() > 0:
                unique_labels, counts = np.unique(ocean_labels[ocean_labels > 0], return_counts=True)
                if len(unique_labels) > 0:
                    largest_label = unique_labels[np.argmax(counts)]
                    ocean_mask = (ocean_labels == largest_label)

        # Combine rock into land (for compatibility)
        land_mask = land_mask | rock_mask

        # Ensure no overlap
        land_mask = land_mask & ~ocean_mask & ~wave_mask

        return {
            'ocean': ocean_mask,
            'land': land_mask,
            'waves': wave_mask
        }


def create_default_prompts(image, output_path="prompts/default_coastal.json"):
    """
    Helper to create default prompts for coastal imagery

    Args:
        image: Sample RGB image to base prompts on
        output_path: Where to save prompts
    """
    h, w = image.shape[:2]

    # Default prompts for typical coastal drone imagery
    # Ocean usually in lower-right area, land in upper-left
    prompts = {
        "ocean": {
            "points": [
                [w * 3//4, h * 3//4],  # Lower right
                [w * 2//3, h * 2//3],  # Mid right
                [w * 5//6, h * 5//6],  # Far lower right
            ],
            "labels": [1, 1, 1]
        },
        "land": {
            "points": [
                [w // 4, h // 4],      # Upper left
                [w // 3, h // 3],      # Mid left
                [w // 6, h // 6],      # Far upper left
            ],
            "labels": [1, 1, 1]
        },
        "rock": {
            "points": [
                [w // 2, h // 2],      # Center (often rocky shore)
            ],
            "labels": [1]
        }
    }

    # Save prompts
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        json.dump(prompts, f, indent=2)

    print(f"Created default prompts: {output_path}")
    print("These are generic - create custom prompts with:")
    print("  python scripts/sam_segmenter.py --interactive --data your_data/")

    return prompts
