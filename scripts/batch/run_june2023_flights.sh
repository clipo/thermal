#!/bin/bash
# Run the spread SGD detector on every June 2023 combined directory.
# Mirrors run_all_flights.sh but iterates the existing data/june2023_*_combined
# pools that build_june2023_extents.py already symlinked.

set -e
THERMAL="/Users/clipo/PycharmProjects/thermal"
LOG_DIR="/tmp/sgd_batch_logs_june2023"
mkdir -p "$LOG_DIR"

cd "$THERMAL"

for combined in data/june2023_*_combined; do
  [ -d "$combined" ] || continue
  slug=$(basename "$combined" | sed 's/_combined$//')
  outdir="sgd_output/${slug}_spread"
  # Resume option: skip flights with existing CSV when SGD_RESUME=1 in env.
  # Default: always re-run (so schema upgrades to existing outputs propagate).
  if [ "${SGD_RESUME:-0}" = "1" ] && [ -f "${outdir}/${slug}_sgd.csv" ]; then
    echo "  ✓ already done: $slug — skipping (SGD_RESUME=1)"
    continue
  fi
  mkdir -p "$outdir"

  echo
  echo "========================================="
  echo "Processing: $slug"
  echo "========================================="

  n_frames=$(ls "$combined" | grep -c "^MAX_.*\.JPG$")
  if [ "$n_frames" -lt 10 ]; then
    echo "  too few frames ($n_frames), skipping"
    continue
  fi

  first=$(ls "$combined" | grep "^MAX_.*\.JPG$" | head -1 | sed 's/MAX_//; s/.JPG//')
  last=$(ls "$combined" | grep "^MAX_.*\.JPG$" | tail -1 | sed 's/MAX_//; s/.JPG//')
  echo "  combined: $combined ($n_frames paired frames, range $first - $last)"

  log="$LOG_DIR/${slug}.log"

  python -u scripts/pipeline/run_coast_stretch.py \
    --detector spread \
    --data "$combined" \
    --start "$((10#$first))" --end "$((10#$last))" \
    --save-raw \
    --local-adaptive-fraction 0.85 \
    --output "${outdir}/${slug}_sgd" 2>&1 | tee "$log" \
    | grep -E "Processing|raw polys|Clustering|merged polygons|total merged|elapsed|local-adaptive"
done

echo
echo "JUNE 2023 ALL FLIGHTS DONE"
echo "Results in: $THERMAL/sgd_output/june2023_*_spread/"
