#!/usr/bin/env python3
"""Walk the June 2023 thermal survey tree and build per-flight extents.

The June 2023 directory layout differs from Jan 2024:
    Thermal Flights/<date>/[<named flight>/]<NNNMEDIA>/MAX_*.JPG

Some dates have named flight subdirectories (e.g., "Hanga Roa - Rano Kau",
"Poike 1") and others have MEDIA folders directly under the date. We treat
each named subdirectory as a distinct flight; flat dates become one flight
named after the date.

For each identified flight: symlink all MAX/IRX pairs from its MEDIA folders
into `data/june2023_<slug>_combined/`, then run ThermalFrameMapper to produce
extent KML in `sgd_output/june2023_<slug>_extents/`.

Usage:
    python scripts/build_june2023_extents.py
    python scripts/build_june2023_extents.py --frame-skip 10  # faster
    python scripts/build_june2023_extents.py --filter "Vaihu"  # subset
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from sgd_toolkit.georeferencing.footprint_generator import ThermalFrameMapper

JUNE_BASE = Path("/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights")


def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def find_media_dirs(root: Path) -> list[Path]:
    """Find subdirectories ending in MEDIA or FTASK that contain MAX_*.JPG files."""
    out = []
    for sub in root.rglob("*"):
        if not sub.is_dir():
            continue
        name = sub.name
        if not (name.endswith("MEDIA") or name.endswith("FTASK")):
            continue
        # confirm has MAX files
        try:
            for f in sub.iterdir():
                if f.name.startswith("MAX_"):
                    out.append(sub)
                    break
        except Exception:
            continue
    return sorted(out)


def discover_flights() -> list[tuple[str, list[Path]]]:
    """Return list of (flight_label, [media_dirs])."""
    flights: list[tuple[str, list[Path]]] = []
    for date_dir in sorted(JUNE_BASE.iterdir()):
        if not date_dir.is_dir():
            continue
        if not re.match(r"\d", date_dir.name):
            # skip non-date folders like "Coast Photos", "extracts", etc.
            continue
        media_at_date = [d for d in date_dir.iterdir() if d.is_dir() and (d.name.endswith("MEDIA") or d.name.endswith("FTASK"))]
        named_subs = [d for d in date_dir.iterdir() if d.is_dir() and not (d.name.endswith("MEDIA") or d.name.endswith("FTASK"))]

        # Filter named subs to only those that actually contain MEDIA folders
        named_with_media = []
        for sub in named_subs:
            if find_media_dirs(sub):
                named_with_media.append(sub)

        if named_with_media:
            for ns in named_with_media:
                flight_label = f"{date_dir.name}/{ns.name}"
                flights.append((flight_label, find_media_dirs(ns)))
            # Skip flat MEDIA at date level when named subs exist — they're
            # almost always the same data referenced through a named container
            # like "SortPhotos" or other organizing folder.
        elif media_at_date:
            flights.append((date_dir.name, media_at_date))
    return flights


def prepare_combined_dir(flight_label: str, media_dirs: list[Path]) -> tuple[Path, int]:
    slug = "june2023_" + slugify(flight_label)
    combined = Path("data") / f"{slug}_combined"
    combined.mkdir(parents=True, exist_ok=True)
    # Clean stale symlinks
    for p in combined.iterdir():
        if p.is_symlink():
            p.unlink()

    n = 0
    for md in media_dirs:
        for f in md.iterdir():
            if not f.is_file():
                continue
            target = combined / f.name
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(f)
            if f.name.startswith("MAX_"):
                n += 1
    return combined, n


def build_extent(flight_label: str, combined: Path, output_root: Path, frame_skip: int) -> dict:
    slug = "june2023_" + slugify(flight_label)
    out_dir = output_root / f"{slug}_extents"
    out_dir.mkdir(parents=True, exist_ok=True)

    mapper = ThermalFrameMapper(
        data_dir=str(combined),
        output_base=f"{slug}_extent",
        frame_skip=frame_skip,
        verbose=False,
    )
    mapper.output_dir = out_dir
    try:
        ok = mapper.run()
    except Exception as e:
        return {"slug": slug, "label": flight_label, "ok": False, "reason": f"crash: {e}", "n_frames": 0}
    return {
        "slug": slug,
        "label": flight_label,
        "ok": ok,
        "n_frames": len(getattr(mapper, "frame_footprints", [])),
        "merged_kml": str(out_dir / f"{slug}_extent_merged.kml"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", default="sgd_output")
    ap.add_argument("--frame-skip", type=int, default=10)
    ap.add_argument("--filter", default=None, help="Only process flight labels containing this substring (case-insensitive)")
    ap.add_argument("--dry-run", action="store_true", help="List what would be processed without doing it")
    args = ap.parse_args()

    flights = discover_flights()
    if args.filter:
        f = args.filter.lower()
        flights = [fl for fl in flights if f in fl[0].lower()]
    print(f"Discovered {len(flights)} flights:")
    for label, mds in flights:
        n = sum(sum(1 for x in md.iterdir() if x.name.startswith("MAX_")) for md in mds)
        print(f"  {label} — {len(mds)} media dirs, {n} frames")
    if args.dry_run:
        return

    output_root = Path(args.output_root)
    results = []
    for label, mds in flights:
        print(f"\n=== {label} ===")
        try:
            combined, n = prepare_combined_dir(label, mds)
            print(f"  combined: {combined} ({n} frames symlinked)")
            r = build_extent(label, combined, output_root, args.frame_skip)
        except Exception as e:
            r = {"slug": "?", "label": label, "ok": False, "reason": f"setup crash: {e}", "n_frames": 0}
        if r.get("ok"):
            print(f"  ✓ {r['n_frames']} frames mapped → {r['merged_kml']}")
        else:
            print(f"  ✗ failed: {r.get('reason', 'unknown')}")
        results.append(r)

    print("\n--- summary ---")
    for r in results:
        status = "OK" if r.get("ok") else "FAIL"
        print(f"  {status:>4}  {r['label']:<60}  {r.get('n_frames', 0):>5} frames")
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{n_ok}/{len(results)} flights succeeded")


if __name__ == "__main__":
    main()
