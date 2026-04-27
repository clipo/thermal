"""
False-positive filters for SGD detection candidates.

Each filter exports one function:

    filter(candidate_mask, rgb, thermal, meta, **kwargs) ->
        (kept_mask, rejection_reasons)

where:
    candidate_mask: bool array of per-pixel plume candidates
    rgb: HxWx3 uint8 RGB (aligned to thermal)
    thermal: HxW float °C
    meta: dict with at least {'timestamp', 'gps_lat', 'gps_lon'} when available
    kept_mask: bool array, candidate pixels not flagged by this filter
    rejection_reasons: dict[int, str] keyed by connected-component label

Filters compose by intersecting their kept_masks. Rejection reasons are
collected across filters so the viewer can show why each candidate was
dropped.
"""

from sgd_toolkit.filters.rock_filter import filter_rocks
from sgd_toolkit.filters.shadow_filter import filter_shadows

__all__ = ["filter_rocks", "filter_shadows"]
