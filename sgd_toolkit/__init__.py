"""
SGD Toolkit - Submarine Groundwater Discharge Detection from Thermal Imagery

A comprehensive toolkit for detecting and analyzing submarine groundwater discharge
using thermal and RGB imagery from UAV surveys.
"""

__version__ = "1.0.0"
__author__ = "Thermal SGD Detection Team"

# Import main classes for easy access
from sgd_toolkit.detectors.base import IntegratedSGDDetector
from sgd_toolkit.detectors.improved import ImprovedSGDDetector
from sgd_toolkit.detectors.temporal import MovingAverageSGDDetector
from sgd_toolkit.detectors.edge_aware import EdgeAwareSGDDetector

from sgd_toolkit.segmentation.ml_segmenter import FastMLSegmenter

__all__ = [
    'IntegratedSGDDetector',
    'ImprovedSGDDetector',
    'MovingAverageSGDDetector',
    'EdgeAwareSGDDetector',
    'FastMLSegmenter',
]
