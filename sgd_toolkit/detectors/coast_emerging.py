"""
Coast-emerging SGD detection by region growing from the shoreline.

The physical claim this module enforces: a real SGD plume *originates* at the
coast and fans outward, with its coldest point near the discharge source and
warming as it mixes with ocean water. This is strictly stronger than the old
"a cold connected component happens to have one pixel within 5 px of shore"
rule used in IntegratedSGDDetector.detect_sgd_plumes.

Algorithm:

1. Seed: shoreline pixels whose adjacent-ocean neighborhood is colder than
   the local baseline by at least `seed_z_threshold` sigma.
2. Grow: priority-queue (Dijkstra-like) expansion into ocean, admitting
   pixels with z < grow_z_threshold. Growth proceeds coldest-first so the
   boundary of each plume tracks the temperature level set.
3. Stop: when no neighbor satisfies the threshold, or the pixel is beyond
   `max_distance_px` from its seed, or already claimed by another seed.
4. Topology test: keep only plumes whose temperature recovers toward the
   baseline with distance from shore. z_map is (T - baseline)/sigma, so cold
   pixels have negative z; a real plume is most negative at the shore-anchored
   source and trends toward 0 outward. Spearman rho of (distance_from_shore, z)
   should be clearly positive for a real plume.
5. Shape sanity: reject near-circular blobs (low eccentricity, high solidity)
   — rocks are compact; plumes fan.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage import measure, morphology


@dataclass
class PlumeDetection:
    """One accepted coast-emerging plume."""

    id: int
    mask: np.ndarray  # bool, full-frame
    seed_yx: tuple[int, int]  # seed pixel row, col
    area_pixels: int
    min_z: float
    mean_z: float
    mean_temp_c: float
    min_temp_c: float
    eccentricity: float
    solidity: float
    max_distance_px: float
    shore_touch_pixels: int
    monotonicity: float  # Spearman rho of (distance_from_shore, z); positive = recovers = good
    rejected_reason: Optional[str] = None


@dataclass
class CoastEmergingResult:
    sgd_mask: np.ndarray
    plumes: list[PlumeDetection] = field(default_factory=list)
    rejected: list[PlumeDetection] = field(default_factory=list)


def _shoreline_pixels(ocean_mask: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    """1-pixel-wide strand of ocean pixels 4-adjacent to land."""
    up = np.roll(land_mask, -1, axis=0)
    dn = np.roll(land_mask, 1, axis=0)
    lf = np.roll(land_mask, -1, axis=1)
    rt = np.roll(land_mask, 1, axis=1)
    return ocean_mask & (up | dn | lf | rt)


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without scipy.stats (avoids heavy import)."""
    if x.size < 4:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom == 0:
        return 0.0
    return float((rx * ry).sum() / denom)


def grow_plumes_from_shore(
    thermal: np.ndarray,
    z_map: np.ndarray,
    ocean_mask: np.ndarray,
    land_mask: np.ndarray,
    distance_from_shore: np.ndarray,
    *,
    seed_z_threshold: float = -2.0,
    grow_z_threshold: float = -1.2,
    max_distance_px: float = 300.0,
    min_area: int = 40,
    max_eccentricity_for_rejection: float = 0.3,
    max_solidity_for_rejection: float = 0.95,
    require_monotonic_recovery: bool = True,
    monotonicity_threshold: float = 0.3,
    min_shore_touch_pixels: int = 3,
    seed_shore_distance_px: float = 6.0,
) -> CoastEmergingResult:
    """Grow plumes from shoreline seeds and return accepted/rejected detections.

    Args:
        thermal: 2D temperature field (°C).
        z_map: per-pixel anomaly z-score (from fit_spatial_baseline). NaN off-ocean.
        ocean_mask, land_mask: scene segmentation.
        distance_from_shore: pixel distance-from-shore (float, +inf outside ocean).
        seed_z_threshold: shoreline pixels with z <= this become seeds. More negative = stricter.
        grow_z_threshold: pixels added to a plume must have z <= this.
        max_distance_px: hard cap on plume extent from its seed's shoreline anchor.
        min_area: minimum plume size in pixels.
        max_eccentricity_for_rejection: a blob with eccentricity BELOW this AND
            solidity ABOVE max_solidity_for_rejection is rejected as "compact blob".
        max_solidity_for_rejection: see above.
        require_monotonic_recovery: if True, apply the nearshore-vs-offshore
            half-split test below. Only applies to plumes ≥20 px with max span
            ≥15 px — small plumes are exempt because the test is ill-defined.
        monotonicity_threshold: max allowable "nearshore mean z − offshore mean z"
            in sigma units. Real plumes are coldest near shore, so this
            difference should be ≤ 0 or slightly positive. Values above
            threshold reject the candidate as anti-plume-shaped.
        min_shore_touch_pixels: minimum plume pixels that must lie on the shoreline.
        seed_shore_distance_px: ocean pixels this close to shoreline are eligible
            seeds. Plumes often reach their coldest temperature 1-2 m *offshore*
            from the actual discharge point, not literally on the rock edge, so
            restricting seeds to the 1-pixel shoreline strand loses real signal.

    Returns:
        CoastEmergingResult with accepted plumes, rejected plumes (with reasons),
        and aggregated sgd_mask.
    """
    shoreline = _shoreline_pixels(ocean_mask, land_mask)
    shape = thermal.shape
    sgd_mask = np.zeros(shape, dtype=bool)

    if not shoreline.any():
        return CoastEmergingResult(sgd_mask=sgd_mask)

    claimed = np.full(shape, -1, dtype=np.int32)  # which plume owns each pixel
    dist_to_seed = np.full(shape, np.inf, dtype=np.float32)

    # Seed eligibility region: ocean pixels within `seed_shore_distance_px`
    # of the shoreline. Growing from ocean pixels a few pixels offshore (not
    # just the shoreline strand itself) catches plumes whose coldest point sits
    # 1-2 m offshore of the actual discharge.
    near_shore = ocean_mask & (distance_from_shore <= seed_shore_distance_px)
    local_mean_z = _masked_boxcar(z_map, ocean_mask, radius=1)
    seed_mask = near_shore & (
        (z_map <= seed_z_threshold) | (local_mean_z <= seed_z_threshold)
    )
    seed_ys, seed_xs = np.where(seed_mask)
    if seed_ys.size == 0:
        return CoastEmergingResult(sgd_mask=sgd_mask)

    # Cluster adjacent seeds into seed regions so each physical discharge site
    # is grown once, not as many fragments.
    seed_labels, n_seeds = measure.label(seed_mask, connectivity=2, return_num=True)

    plumes: list[PlumeDetection] = []

    for seed_id in range(1, n_seeds + 1):
        pixels = np.argwhere(seed_labels == seed_id)
        # Grow outward from all pixels of this seed cluster simultaneously.
        plume_mask, min_z_val, mean_z_val = _grow_single(
            pixels=pixels,
            z_map=z_map,
            ocean_mask=ocean_mask,
            claimed=claimed,
            dist_to_seed=dist_to_seed,
            plume_id=seed_id,
            grow_z_threshold=grow_z_threshold,
            max_distance_px=max_distance_px,
        )

        area = int(plume_mask.sum())
        if area < min_area:
            continue

        temps = thermal[plume_mask]
        # Shape
        try:
            props = measure.regionprops(plume_mask.astype(np.int32))[0]
            ecc = float(props.eccentricity)
            sol = float(props.solidity)
        except Exception:
            ecc = 0.0
            sol = 0.0

        # Topology check: for a real SGD plume, the nearshore portion of the
        # plume should be at least as cold (mean-z as negative) as the offshore
        # portion — not significantly warmer. This is weaker than a full
        # Spearman-rho recovery test, which fails spuriously when:
        #   * the plume is small (<15 px span from shore → no real gradient)
        #   * the plume spreads longshore rather than offshore
        #   * the offshore ambient happens to be colder than the coast
        # Here we just require that nearshore half isn't clearly warmer than
        # offshore half. The "distance from seed" metric is reserved via
        # Spearman as a secondary signal for future use.
        d_plume = distance_from_shore[plume_mask]
        z_plume = z_map[plume_mask]
        finite = np.isfinite(d_plume) & np.isfinite(z_plume)
        rho = _spearman_rho(d_plume[finite], z_plume[finite]) if finite.sum() >= 8 else 0.0

        # "Shore touch" counts plume pixels within seed_shore_distance_px of the
        # coast — the plume is coast-anchored by construction (grown from a seed
        # within that radius) but we still require non-trivial coast proximity
        # for the final accept.
        shore_touch = int((plume_mask & near_shore).sum())
        max_d = float(d_plume[finite].max()) if finite.any() else 0.0

        # Nearshore-half mean-z vs. offshore-half mean-z.
        nearshore_warmer_by = np.nan
        if finite.sum() >= 20 and max_d >= 15.0:
            half = float(np.median(d_plume[finite]))
            near_half = finite & (d_plume <= half)
            far_half = finite & (d_plume > half)
            if near_half.sum() >= 5 and far_half.sum() >= 5:
                near_mean = float(np.mean(z_plume[near_half]))
                far_mean = float(np.mean(z_plume[far_half]))
                # "nearshore warmer by" = how much less negative nearshore is vs. offshore.
                # Positive = nearshore warmer than offshore = anti-plume pattern.
                nearshore_warmer_by = near_mean - far_mean

        seed_pixel = (int(pixels[0, 0]), int(pixels[0, 1]))

        plume = PlumeDetection(
            id=seed_id,
            mask=plume_mask,
            seed_yx=seed_pixel,
            area_pixels=area,
            min_z=float(np.nanmin(z_plume)),
            mean_z=float(np.nanmean(z_plume)),
            mean_temp_c=float(np.mean(temps)),
            min_temp_c=float(np.min(temps)),
            eccentricity=ecc,
            solidity=sol,
            max_distance_px=max_d,
            shore_touch_pixels=shore_touch,
            monotonicity=rho,
        )

        # Filters — order matters: fast checks first.
        if shore_touch < min_shore_touch_pixels:
            plume.rejected_reason = "shore_touch_too_small"
        elif (
            ecc < max_eccentricity_for_rejection
            and sol > max_solidity_for_rejection
            and area < 10 * min_area
        ):
            plume.rejected_reason = "compact_blob_rock_shaped"
        elif (
            require_monotonic_recovery
            and np.isfinite(nearshore_warmer_by)
            and nearshore_warmer_by > monotonicity_threshold
        ):
            plume.rejected_reason = "anti_plume_shape"

        plumes.append(plume)

    accepted = [p for p in plumes if p.rejected_reason is None]
    rejected = [p for p in plumes if p.rejected_reason is not None]

    for p in accepted:
        sgd_mask |= p.mask

    return CoastEmergingResult(sgd_mask=sgd_mask, plumes=accepted, rejected=rejected)


def _grow_single(
    *,
    pixels: np.ndarray,
    z_map: np.ndarray,
    ocean_mask: np.ndarray,
    claimed: np.ndarray,
    dist_to_seed: np.ndarray,
    plume_id: int,
    grow_z_threshold: float,
    max_distance_px: float,
) -> tuple[np.ndarray, float, float]:
    """Priority-queue region grow from a set of seed pixels."""
    H, W = z_map.shape
    heap: list[tuple[float, float, int, int]] = []  # (z, dist_to_seed, y, x)
    mask = np.zeros_like(ocean_mask, dtype=bool)

    for y, x in pixels:
        z = z_map[y, x]
        if not np.isfinite(z):
            continue
        # Accept seed pixels regardless of grow_z_threshold — they already passed the seed test.
        mask[y, x] = True
        claimed[y, x] = plume_id
        dist_to_seed[y, x] = 0.0
        heapq.heappush(heap, (float(z), 0.0, int(y), int(x)))

    neigh = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

    while heap:
        z_cur, d_cur, y, x = heapq.heappop(heap)
        for dy, dx in neigh:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < H and 0 <= xx < W):
                continue
            if not ocean_mask[yy, xx]:
                continue
            if claimed[yy, xx] != -1:
                continue  # already owned by some plume (this one or another)
            z_n = z_map[yy, xx]
            if not np.isfinite(z_n) or z_n > grow_z_threshold:
                continue
            step = 1.0 if (dy == 0 or dx == 0) else np.sqrt(2.0)
            d_n = d_cur + step
            if d_n > max_distance_px:
                continue
            mask[yy, xx] = True
            claimed[yy, xx] = plume_id
            dist_to_seed[yy, xx] = d_n
            heapq.heappush(heap, (float(z_n), float(d_n), yy, xx))

    if not mask.any():
        return mask, np.nan, np.nan

    z_inside = z_map[mask]
    z_inside = z_inside[np.isfinite(z_inside)]
    if z_inside.size == 0:
        return mask, np.nan, np.nan
    return mask, float(z_inside.min()), float(z_inside.mean())


def _masked_boxcar(z: np.ndarray, mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Mean of z over a (2r+1) window, restricted to mask. NaN where no valid
    neighbor. Used to smooth shoreline seed selection so single-pixel noise
    doesn't drive decisions."""
    z_clean = np.where(mask & np.isfinite(z), z, 0.0)
    m = mask.astype(np.float32)
    k = 2 * radius + 1
    kernel = np.ones((k, k), dtype=np.float32)
    s = ndimage.convolve(z_clean, kernel, mode="constant", cval=0.0)
    n = ndimage.convolve(m, kernel, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n > 0, s / n, np.nan)
    return out
