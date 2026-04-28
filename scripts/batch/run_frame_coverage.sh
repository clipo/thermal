#!/bin/bash

# Script to generate thermal frame coverage KML files
#
# Usage:
#   ./run_frame_coverage.sh /path/to/data
#   ./run_frame_coverage.sh /path/to/data output_name
#   ./run_frame_coverage.sh /path/to/data output_name 5

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_directory> [output_name] [frame_skip]"
    echo ""
    echo "Arguments:"
    echo "  data_directory - Path to folder with MAX_*.JPG and IRX_*.irg files"
    echo "  output_name    - Base name for output files (default: thermal_coverage)"
    echo "  frame_skip     - Process every Nth frame (default: 1 = all frames)"
    echo ""
    echo "Examples:"
    echo "  $0 /Volumes/RapaNui/data"
    echo "  $0 /Volumes/RapaNui/data vaihu_coverage"
    echo "  $0 /Volumes/RapaNui/data vaihu_coverage 5"
    echo ""
    echo "Output:"
    echo "  Creates two KML files in sgd_output/:"
    echo "  - <output_name>_frames.kml - Individual frame outlines"
    echo "  - <output_name>_merged.kml - Combined coverage area"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_NAME="${2:-thermal_coverage}"
FRAME_SKIP="${3:-1}"

echo "========================================"
echo "Thermal Frame Coverage Mapping"
echo "========================================"
echo "Data directory: $DATA_DIR"
echo "Output name: $OUTPUT_NAME"
echo "Frame skip: $FRAME_SKIP (process every ${FRAME_SKIP} frame(s))"
echo ""

# Run the frame footprint generator
python generate_frame_footprints.py \
    --data "$DATA_DIR" \
    --output "$OUTPUT_NAME" \
    --skip "$FRAME_SKIP"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Coverage mapping complete!"
    echo ""
    echo "View the KML files in Google Earth:"
    echo "  - Individual frames: sgd_output/${OUTPUT_NAME}_frames.kml"
    echo "  - Merged coverage: sgd_output/${OUTPUT_NAME}_merged.kml"
else
    echo ""
    echo "❌ Error generating coverage maps"
    exit 1
fi