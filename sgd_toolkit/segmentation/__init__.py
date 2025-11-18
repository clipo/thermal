"""
Ocean/Land Segmentation

Machine learning-based segmentation for classifying pixels as ocean, land, rock, or wave.
Supports both Random Forest (FastMLSegmenter) and SAM (SAMSegmenter).
"""

from sgd_toolkit.segmentation.ml_segmenter import FastMLSegmenter

# SAM segmenter (optional - requires SAM installation)
try:
    from sgd_toolkit.segmentation.sam_segmenter import SAMSegmenter
    __all__ = ['FastMLSegmenter', 'SAMSegmenter']
except ImportError:
    __all__ = ['FastMLSegmenter']
