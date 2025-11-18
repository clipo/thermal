#!/usr/bin/env python3
"""
Sun Glint Detection for SGD Analysis

Detects and filters frames affected by sun glint, which can create
false SGD detections during rapid drone turns or certain sun angles.

Sun glint characteristics:
- Large area of anomalously cold temperatures
- High brightness/saturation in RGB
- Sudden appearance/disappearance between frames
- Often affects significant portion of ocean area
"""

import numpy as np
from pathlib import Path
from skimage import morphology, measure
from scipy import stats
from skimage import color


class SunGlintDetector:
    """Detect sun glint in thermal/RGB drone imagery"""

    def __init__(self,
                 brightness_threshold=0.8,  # Fraction of max brightness
                 area_threshold=0.2,  # Fraction of ocean area affected
                 temp_std_threshold=3.0,  # Standard deviations from mean
                 frame_discontinuity_threshold=2.0):  # °C difference from adjacent frames
        """
        Initialize sun glint detector

        Args:
            brightness_threshold: RGB brightness threshold (0-1)
            area_threshold: Minimum fraction of ocean affected to flag as glint
            temp_std_threshold: Temperature standard deviations to flag anomaly
            frame_discontinuity_threshold: Temperature jump between frames
        """
        self.brightness_threshold = brightness_threshold
        self.area_threshold = area_threshold
        self.temp_std_threshold = temp_std_threshold
        self.frame_discontinuity_threshold = frame_discontinuity_threshold

        # Store frame history for continuity checks
        self.frame_history = []
        self.max_history = 5  # Keep last 5 frames

    def detect_glint_rgb(self, rgb_image, ocean_mask):
        """
        Detect sun glint based on RGB brightness patterns

        Args:
            rgb_image: RGB image (H x W x 3)
            ocean_mask: Boolean mask of ocean areas

        Returns:
            dict: Glint detection metrics
        """
        # Convert to grayscale for brightness analysis
        if len(rgb_image.shape) == 3:
            gray = color.rgb2gray(rgb_image)
        else:
            gray = rgb_image

        # Normalize to 0-1
        if gray.max() > 1:
            gray = gray.astype(float) / 255.0

        # Focus on ocean areas
        ocean_brightness = gray[ocean_mask]

        if len(ocean_brightness) == 0:
            return {
                'has_glint': False,
                'reason': 'No ocean area'
            }

        # Check for high brightness areas
        bright_threshold = self.brightness_threshold
        bright_pixels = ocean_brightness > bright_threshold
        bright_fraction = np.sum(bright_pixels) / len(ocean_brightness)

        # Check for saturation (near white)
        saturated = ocean_brightness > 0.95
        saturated_fraction = np.sum(saturated) / len(ocean_brightness)

        # Calculate brightness statistics
        brightness_mean = np.mean(ocean_brightness)
        brightness_std = np.std(ocean_brightness)
        brightness_max = np.max(ocean_brightness)

        # Detect glint based on brightness patterns
        has_glint = (
            bright_fraction > self.area_threshold or
            saturated_fraction > 0.1 or  # >10% saturated pixels
            brightness_mean > 0.7  # Overall very bright
        )

        return {
            'has_glint': has_glint,
            'bright_fraction': bright_fraction,
            'saturated_fraction': saturated_fraction,
            'brightness_mean': brightness_mean,
            'brightness_std': brightness_std,
            'brightness_max': brightness_max,
            'reason': 'High brightness/saturation' if has_glint else 'Normal brightness'
        }

    def detect_glint_thermal(self, thermal_image, ocean_mask, sgd_mask=None):
        """
        Detect sun glint based on thermal patterns

        Args:
            thermal_image: Thermal image (temperatures)
            ocean_mask: Boolean mask of ocean areas
            sgd_mask: Optional mask of detected SGD areas

        Returns:
            dict: Glint detection metrics
        """
        ocean_temps = thermal_image[ocean_mask]

        if len(ocean_temps) == 0:
            return {
                'has_glint': False,
                'reason': 'No ocean area'
            }

        # Calculate robust statistics
        temp_median = np.median(ocean_temps)
        temp_mad = stats.median_abs_deviation(ocean_temps)
        temp_std = np.std(ocean_temps)

        # Check if SGD detection covers unusually large area
        if sgd_mask is not None:
            ocean_pixels = np.sum(ocean_mask)
            sgd_pixels = np.sum(sgd_mask)
            sgd_fraction = sgd_pixels / ocean_pixels if ocean_pixels > 0 else 0

            # Flag if "SGD" covers too much area (likely glint)
            if sgd_fraction > self.area_threshold:
                return {
                    'has_glint': True,
                    'sgd_fraction': sgd_fraction,
                    'temp_median': temp_median,
                    'temp_std': temp_std,
                    'reason': f'SGD area too large ({sgd_fraction:.1%} of ocean)'
                }

        # Check for unusual temperature distribution
        # Sun glint often creates uniform cold areas
        cold_threshold = temp_median - self.temp_std_threshold * temp_mad
        cold_pixels = ocean_temps < cold_threshold
        cold_fraction = np.sum(cold_pixels) / len(ocean_temps)

        # Check spatial coherence of cold areas
        if sgd_mask is not None:
            # Label connected components
            labeled = measure.label(sgd_mask, connectivity=2)
            if labeled.max() > 0:
                # Get largest component
                props = measure.regionprops(labeled)
                if props:
                    largest = max(props, key=lambda x: x.area)

                    # Check if it's too regular/circular (glint tends to be)
                    if largest.eccentricity < 0.5 and largest.area > 1000:
                        return {
                            'has_glint': True,
                            'cold_fraction': cold_fraction,
                            'largest_area': largest.area,
                            'eccentricity': largest.eccentricity,
                            'reason': 'Large circular cold anomaly (likely glint)'
                        }

        return {
            'has_glint': cold_fraction > self.area_threshold,
            'cold_fraction': cold_fraction,
            'temp_median': temp_median,
            'temp_std': temp_std,
            'reason': 'Large cold area detected' if cold_fraction > self.area_threshold else 'Normal thermal'
        }

    def check_frame_continuity(self, thermal_image, ocean_mask):
        """
        Check for sudden temperature changes between frames

        Args:
            thermal_image: Current thermal image
            ocean_mask: Boolean mask of ocean areas

        Returns:
            dict: Continuity check results
        """
        ocean_temps = thermal_image[ocean_mask]

        if len(ocean_temps) == 0:
            return {
                'has_discontinuity': False,
                'reason': 'No ocean area'
            }

        current_median = np.median(ocean_temps)

        # Add to history
        self.frame_history.append(current_median)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)

        # Need at least 2 frames for comparison
        if len(self.frame_history) < 2:
            return {
                'has_discontinuity': False,
                'reason': 'Insufficient frame history'
            }

        # Check for sudden temperature jumps
        prev_median = self.frame_history[-2]
        temp_jump = abs(current_median - prev_median)

        # Also check against running average
        if len(self.frame_history) >= 3:
            running_avg = np.mean(self.frame_history[:-1])
            deviation = abs(current_median - running_avg)

            has_discontinuity = (
                temp_jump > self.frame_discontinuity_threshold or
                deviation > self.frame_discontinuity_threshold * 1.5
            )

            return {
                'has_discontinuity': has_discontinuity,
                'temp_jump': temp_jump,
                'deviation': deviation,
                'current_median': current_median,
                'prev_median': prev_median,
                'running_avg': running_avg,
                'reason': f'Temperature jump: {temp_jump:.1f}°C' if has_discontinuity else 'Continuous'
            }

        return {
            'has_discontinuity': temp_jump > self.frame_discontinuity_threshold,
            'temp_jump': temp_jump,
            'current_median': current_median,
            'prev_median': prev_median,
            'reason': f'Temperature jump: {temp_jump:.1f}°C' if temp_jump > self.frame_discontinuity_threshold else 'Continuous'
        }

    def detect_turn_glint(self, rgb_image, thermal_image, ocean_mask, sgd_mask=None,
                         check_continuity=True, heading_change=None):
        """
        Comprehensive sun glint detection during turns

        Args:
            rgb_image: RGB image
            thermal_image: Thermal image
            ocean_mask: Boolean mask of ocean areas
            sgd_mask: Optional mask of detected SGD areas
            check_continuity: Whether to check frame continuity
            heading_change: Optional heading change in degrees

        Returns:
            dict: Comprehensive glint detection results
        """
        results = {
            'has_glint': False,
            'confidence': 0.0,
            'reasons': []
        }

        # Check RGB brightness patterns
        rgb_glint = self.detect_glint_rgb(rgb_image, ocean_mask)
        if rgb_glint['has_glint']:
            results['reasons'].append(f"RGB: {rgb_glint['reason']}")
            results['confidence'] += 0.4
        results['rgb_analysis'] = rgb_glint

        # Check thermal patterns
        thermal_glint = self.detect_glint_thermal(thermal_image, ocean_mask, sgd_mask)
        if thermal_glint['has_glint']:
            results['reasons'].append(f"Thermal: {thermal_glint['reason']}")
            results['confidence'] += 0.3
        results['thermal_analysis'] = thermal_glint

        # Check frame continuity
        if check_continuity:
            continuity = self.check_frame_continuity(thermal_image, ocean_mask)
            if continuity['has_discontinuity']:
                results['reasons'].append(f"Continuity: {continuity['reason']}")
                results['confidence'] += 0.2
            results['continuity_analysis'] = continuity

        # Check heading change (rapid turns increase glint likelihood)
        if heading_change is not None and abs(heading_change) > 15:  # >15 degree turn
            results['reasons'].append(f"Rapid turn: {heading_change:.1f}°")
            results['confidence'] += 0.1
            results['heading_change'] = heading_change

        # Determine if frame has glint based on combined evidence
        results['has_glint'] = results['confidence'] >= 0.5

        if results['has_glint']:
            results['summary'] = f"Sun glint detected (confidence: {results['confidence']:.0%})"
        else:
            results['summary'] = "No sun glint detected"

        return results

    def filter_glint_frames(self, frame_results):
        """
        Filter out frames affected by sun glint

        Args:
            frame_results: List of frame analysis results

        Returns:
            list: Filtered results without glint frames
        """
        filtered = []

        for result in frame_results:
            if 'glint_analysis' in result:
                if not result['glint_analysis']['has_glint']:
                    filtered.append(result)
                else:
                    print(f"  Filtered frame {result.get('frame_number', 'unknown')}: "
                          f"{result['glint_analysis']['summary']}")
            else:
                # No glint analysis, include frame
                filtered.append(result)

        return filtered

    def reset_history(self):
        """Reset frame history (use when switching to new flight/area)"""
        self.frame_history = []


def test_glint_detection():
    """Test sun glint detection"""
    print("Sun Glint Detection Test")
    print("=" * 50)

    # Create synthetic test data
    h, w = 512, 640

    # Normal ocean
    normal_ocean = np.ones((h, w)) * 20.0  # 20°C
    normal_ocean += np.random.normal(0, 0.5, (h, w))

    # Ocean with sun glint (large cold area)
    glint_ocean = normal_ocean.copy()
    glint_ocean[100:400, 200:500] -= 5.0  # Large cold patch

    # Ocean mask
    ocean_mask = np.ones((h, w), dtype=bool)
    ocean_mask[:50, :] = False  # Top is land

    # RGB with glint (bright area)
    rgb_normal = np.ones((h, w, 3), dtype=np.uint8) * 100
    rgb_glint = rgb_normal.copy()
    rgb_glint[100:400, 200:500] = 230  # Bright patch

    # Test detector
    detector = SunGlintDetector()

    print("\nTest 1: Normal frame")
    result = detector.detect_turn_glint(
        rgb_normal, normal_ocean, ocean_mask, check_continuity=False
    )
    print(f"  Has glint: {result['has_glint']}")
    print(f"  Confidence: {result['confidence']:.0%}")

    print("\nTest 2: Frame with sun glint")
    result = detector.detect_turn_glint(
        rgb_glint, glint_ocean, ocean_mask, check_continuity=False
    )
    print(f"  Has glint: {result['has_glint']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Reasons: {', '.join(result['reasons'])}")

    print("\nTest 3: Frame continuity check")
    # Process normal frame first
    detector.detect_turn_glint(rgb_normal, normal_ocean, ocean_mask)
    # Then glint frame (should detect discontinuity)
    result = detector.detect_turn_glint(
        rgb_glint, glint_ocean, ocean_mask, check_continuity=True
    )
    print(f"  Has glint: {result['has_glint']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    if 'continuity_analysis' in result:
        print(f"  Temperature jump: {result['continuity_analysis'].get('temp_jump', 0):.1f}°C")

    print("\n✓ Sun glint detection test complete")


if __name__ == "__main__":
    test_glint_detection()