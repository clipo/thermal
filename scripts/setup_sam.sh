#!/bin/bash
# Setup script for Segment Anything Model (SAM)

echo "Setting up SAM for thermal image segmentation..."

# Install additional dependencies first
echo "Installing dependencies..."
# Use compatible opencv-python version with numpy 1.x (for pandas compatibility)
pip install 'opencv-python<4.10.0' 'numpy>=1.26.0,<2.0' matplotlib

# Install SAM
echo "Installing SAM..."
pip install git+https://github.com/facebookresearch/segment-anything.git

# Create models directory if it doesn't exist
mkdir -p models/sam

# Download SAM checkpoint (you'll need to choose a size)
echo ""
echo "SAM Model Options:"
echo "  1. ViT-H (Huge)   - 2.5GB - Best accuracy (recommended for your DGX)"
echo "  2. ViT-L (Large)  - 1.2GB - Good accuracy, faster"
echo "  3. ViT-B (Base)   - 375MB - Fastest, good for testing"
echo ""
echo "Choose model size [1-3]: "
read -r choice

case $choice in
  1)
    echo "Downloading ViT-H (Huge) model..."
    wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ;;
  2)
    echo "Downloading ViT-L (Large) model..."
    wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    ;;
  3)
    echo "Downloading ViT-B (Base) model..."
    wget -P models/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    ;;
  *)
    echo "Invalid choice. Please run again and choose 1, 2, or 3."
    exit 1
    ;;
esac

echo ""
echo "✓ SAM installation complete!"
echo ""
echo "Next steps:"
echo "  1. Test SAM: python scripts/sam_segmenter.py --test"
echo "  2. Create prompts: python scripts/sam_segmenter.py --interactive"
echo "  3. Batch process: python scripts/sam_segmenter.py --data data/100MEDIA"
