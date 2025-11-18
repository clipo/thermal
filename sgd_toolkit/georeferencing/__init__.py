"""
Georeferencing and Coverage Mapping

Tools for converting pixel coordinates to geographic coordinates and generating
coverage footprints for thermal surveys.
"""

from sgd_toolkit.georeferencing.polygon_georef import SGDPolygonGeoref
from sgd_toolkit.georeferencing.footprint_generator import ThermalFrameMapper

__all__ = [
    'SGDPolygonGeoref',
    'ThermalFrameMapper',
]
