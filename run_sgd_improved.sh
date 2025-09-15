#!/bin/bash

# Script to run SGD detection with improved baseline temperature methods
#
# Usage examples:
#   ./run_sgd_improved.sh /path/to/data output.kml
#   ./run_sgd_improved.sh /path/to/data output.kml upper_quartile
#   ./run_sgd_improved.sh /path/to/data output.kml percentile_90

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_directory> <output_kml> [baseline_method]"
    echo ""
    echo "Baseline methods:"
    echo "  median          - Traditional median (default)"
    echo "  upper_quartile  - 75th percentile (recommended for cold-dominated frames)"
    echo "  percentile_80   - 80th percentile"
    echo "  percentile_90   - 90th percentile"
    echo "  trimmed_mean    - Mean after excluding coldest 25%"
    echo ""
    echo "Example:"
    echo "  $0 /Volumes/RapaNui/data survey.kml upper_quartile"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_FILE="$2"
BASELINE_METHOD="${3:-median}"

echo "Running SGD detection with improved baseline method: $BASELINE_METHOD"
echo "Data directory: $DATA_DIR"
echo "Output file: $OUTPUT_FILE"
echo ""

# Run the detection
python sgd_autodetect.py \
    --data "$DATA_DIR" \
    --output "$OUTPUT_FILE" \
    --baseline "$BASELINE_METHOD" \
    --temp 0.5 \
    --area 50 \
    --distance 10

echo ""
echo "Detection complete! Results saved to: $OUTPUT_FILE"