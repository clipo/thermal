#!/bin/bash
# Build cold-anomaly rasters for every flight directory.
# Skips Vaihu East/West (superseded by combined vaihu_full) and Flight 5 (no signal).

set -e
THERMAL="/Users/clipo/PycharmProjects/thermal"
LOG_DIR="/tmp/anomaly_batch_logs"
mkdir -p "$LOG_DIR"

cd "$THERMAL"

# Skip list: superseded or empty flights
SKIP=(
  "vaihu_west_combined"               # superseded by vaihu_full_combined
  "flight4_vaihu_east_full_combined"  # superseded by vaihu_full_combined
  "flight5_vaihu_mission_combined"    # 0 SGD signal
)

for combined in data/*_combined; do
  [ -d "$combined" ] || continue
  name=$(basename "$combined")
  # skip-check
  skip=false
  for s in "${SKIP[@]}"; do
    [ "$name" = "$s" ] && skip=true && break
  done
  if $skip; then
    echo "skip: $name"
    continue
  fi
  slug="${name%_combined}"
  outdir="sgd_output/${slug}_spread"
  mkdir -p "$outdir"

  # Skip if anomaly raster already exists (idempotent)
  if [ -f "${outdir}/${slug}_anomaly.npz" ]; then
    echo "  ✓ already done: $slug"
    continue
  fi

  echo
  echo "==========================="
  echo "Anomaly raster: $slug"
  echo "==========================="

  log="$LOG_DIR/${slug}.log"
  python -u scripts/build_anomaly_raster.py \
    --data "$combined" \
    --output "${outdir}/${slug}_anomaly" \
    --grid-resolution-m 1.0 2>&1 | tee "$log" \
    | grep -E "Pass|frames\.\.\.|wrote|grid|baseline"
done

echo
echo "ANOMALY RASTERS DONE"
echo "Results: sgd_output/*_anomaly.{npz,kml,png}"
