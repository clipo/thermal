#!/bin/bash
# Run the spread SGD detector across every Rapa Nui flight in
# /Volumes/RapaNui/Rapa Nui Jan 2024/Autel/.
#
# Each flight's MEDIA / FTASK subfolders are symlinked into a single combined
# directory under data/<flight_slug>_combined/, then run_coast_stretch.py
# --detector spread is invoked on it. Outputs land in
# sgd_output/<flight_slug>_spread/.

set -e
BASE="/Volumes/RapaNui/Rapa Nui Jan 2024/Autel"
THERMAL="/Users/clipo/PycharmProjects/thermal"
LOG_DIR="/tmp/sgd_batch_logs"
mkdir -p "$LOG_DIR"

# Format: flight_dir_name | output_slug | media_pattern
# media_pattern is a glob relative to the flight dir; we collect all matching
# subdirectories and symlink their contents.
declare -a FLIGHTS=(
  "Flight 1 - West of Tetenga|flight1_west_of_tetenga|*MEDIA"
  "Flight 2  -East of Tetenga|flight2_east_of_tetenga|*MEDIA"
  "Flight 4 - Vaihu - East|flight4_vaihu_east_full|*MEDIA"
  "Flight 5 - Vaihu - Mission|flight5_vaihu_mission|*FTASK"
  "Flight 6 - One Makihi - East|flight6_one_makihi_east|108MEDIA"
  "Flight 7 - Hekii - East|flight7_hekii_east|*MEDIA"
  "Flight 8 - Hekii - West|flight8_hekii_west|*MEDIA"
  "Flight 10 - Anakena to West|flight10_anakena_to_west|*MEDIA"
  "Flight 11 - Hiva Hiva to Hanga Piko|flight11_hivahiva_to_hangapiko|*MEDIA"
)

prepare_combined_dir() {
  local flight_dir="$1"
  local output_slug="$2"
  local media_pattern="$3"
  local combined="$THERMAL/data/${output_slug}_combined"

  mkdir -p "$combined"
  # Clear stale symlinks (keep the dir, just remove old links)
  find "$combined" -maxdepth 1 -type l -delete 2>/dev/null || true

  shopt -s nullglob
  for sub in "$BASE/$flight_dir"/$media_pattern; do
    [ -d "$sub" ] || continue
    for f in "$sub"/*; do
      [ -f "$f" ] || continue
      ln -sf "$f" "$combined/$(basename "$f")"
    done
  done
  shopt -u nullglob

  local n=$(ls "$combined" | grep -c "^MAX_")
  echo "$n"
}

frame_range() {
  local combined="$1"
  local first=$(ls "$combined" | grep "^MAX_" | head -1 | sed 's/MAX_//; s/.JPG//')
  local last=$(ls "$combined" | grep "^MAX_" | tail -1 | sed 's/MAX_//; s/.JPG//')
  echo "$first $last"
}

cd "$THERMAL"

for entry in "${FLIGHTS[@]}"; do
  IFS='|' read -r flight_dir output_slug media_pattern <<< "$entry"
  echo
  echo "========================================="
  echo "Processing: $flight_dir"
  echo "========================================="

  combined="$THERMAL/data/${output_slug}_combined"
  outdir="sgd_output/${output_slug}_spread"
  mkdir -p "$outdir"

  n_frames=$(prepare_combined_dir "$flight_dir" "$output_slug" "$media_pattern")
  echo "  combined dir: $combined  ($n_frames paired frames)"

  if [ "$n_frames" -lt 10 ]; then
    echo "  too few frames, skipping"
    continue
  fi

  read -r first last <<< $(frame_range "$combined")
  echo "  frame range: $first - $last"

  log="$LOG_DIR/${output_slug}.log"
  echo "  log: $log"

  python -u scripts/run_coast_stretch.py \
    --detector spread \
    --data "$combined" \
    --start "$((10#$first))" --end "$((10#$last))" \
    --save-raw \
    --local-adaptive-fraction 0.85 \
    --output "${outdir}/${output_slug}_sgd" 2>&1 | tee "$log" \
    | grep -E "Processing|raw polys|Clustering|merged polygons|total merged|elapsed|local-adaptive"
done

echo
echo "ALL FLIGHTS DONE"
echo "Results in: $THERMAL/sgd_output/*_spread/"
