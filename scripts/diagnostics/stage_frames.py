#!/usr/bin/env python3
"""Copy the frames a diagnostic needs onto local disk before analysing them.

Why
---
The survey frames live on an external volume reached through a symlink farm
under `data/<slug>_combined/`. That volume has proved unreliable during long
analysis runs: it has unmounted outright and has returned `Errno 60` I/O
timeouts under concurrent load, which kills runs partway and produces
truncated, spatially biased samples. Diagnostics that abort cleanly waste an
hour; ones that do not would silently produce a wrong answer.

Copying the sampled frames locally first removes the dependency entirely. The
analysis then reads from local disk at full speed and cannot be interrupted by
the enclosure sleeping.

This copier is RESUMABLE. Files already present with a matching size are
skipped, so if the volume drops mid-copy you can remount and re-run the same
command to continue where it stopped. Nothing is ever deleted from the source.

Usage
-----
    # Stage the block sample the paired / sun tests use
    python scripts/diagnostics/stage_frames.py \\
        --data data/flight4_vaihu_east_full_combined \\
        --dest data/staged/flight4_vaihu_east_full \\
        --n-blocks 8 --block-len 25

    # Then point the diagnostic at the staged copy
    python scripts/diagnostics/radial_paired_test.py \\
        --data data/staged/flight4_vaihu_east_full ...
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


def sample_blocks(nums: list[int], n_blocks: int, block_len: int) -> list[int]:
    """Contiguous runs of frames, evenly spaced across the flight.

    Must match the sampling in radial_paired_test.py and sun_asymmetry_test.py,
    which need consecutive-frame overlap within a block and heading diversity
    across blocks. Staged generously (more blocks / longer blocks than the
    analysis asks for) is safe; staged short is not.
    """
    total = len(nums)
    if n_blocks * block_len >= total:
        return list(nums)
    starts = np.linspace(0, total - block_len, n_blocks).round().astype(int)
    out: list[int] = []
    for s in sorted(set(starts.tolist())):
        out.extend(nums[s:s + block_len])
    return sorted(set(out))


def paired_frames(data_dir: Path) -> list[int]:
    return sorted(
        n for n in (int(p.name[4:8]) for p in data_dir.glob("MAX_*.JPG") if p.name[4:8].isdigit())
        if (data_dir / f"IRX_{n:04d}.irg").exists()
    )


def copy_one(src: Path, dst: Path) -> tuple[bool, int]:
    """Copy src->dst unless dst already matches by size. Returns (copied, bytes)."""
    try:
        s_size = src.stat().st_size
    except OSError as e:
        raise RuntimeError(f"source unreadable: {e}") from e
    if dst.exists() and dst.stat().st_size == s_size and s_size > 0:
        return False, s_size
    tmp = dst.with_suffix(dst.suffix + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)
    return True, s_size


def main():
    ap = argparse.ArgumentParser(
        description="Stage sampled MAX/IRX frame pairs onto local disk. Resumable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True, help="Source dir (the symlink farm)")
    ap.add_argument("--dest", required=True, help="Local destination dir")
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--block-len", type=int, default=25)
    ap.add_argument("--all", action="store_true", help="Stage every frame, not a block sample")
    ap.add_argument("--max-consecutive-failures", type=int, default=10,
                    help="Stop if this many frames in a row fail to copy, so a "
                         "dropped volume ends the run instead of silently "
                         "producing a partial stage.")
    args = ap.parse_args()

    src_dir = Path(args.data)
    dst_dir = Path(args.dest)
    if not src_dir.is_dir():
        raise SystemExit(f"Not a directory: {src_dir}")

    nums = paired_frames(src_dir)
    if not nums:
        raise SystemExit(
            f"No paired MAX/IRX frames visible in {src_dir}. If this is a symlink "
            f"farm, the external volume is not mounted."
        )
    frames = list(nums) if args.all else sample_blocks(nums, args.n_blocks, args.block_len)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"staging {len(frames)} of {len(nums)} frames")
    print(f"  {src_dir}  ->  {dst_dir}")

    copied = skipped = 0
    total_bytes = 0
    consecutive = 0
    failures: list[tuple[int, str]] = []

    for i, n in enumerate(frames, start=1):
        pair = [(src_dir / f"MAX_{n:04d}.JPG", dst_dir / f"MAX_{n:04d}.JPG"),
                (src_dir / f"IRX_{n:04d}.irg", dst_dir / f"IRX_{n:04d}.irg")]
        try:
            for s, d in pair:
                did, nbytes = copy_one(s, d)
                total_bytes += nbytes if did else 0
                if did:
                    copied += 1
                else:
                    skipped += 1
            consecutive = 0
        except Exception as e:
            consecutive += 1
            failures.append((n, f"{type(e).__name__}: {e}"))
            # Drop a half-written pair so a resume redoes it cleanly.
            for _, d in pair:
                if d.with_suffix(d.suffix + ".part").exists():
                    d.with_suffix(d.suffix + ".part").unlink()
            if consecutive >= args.max_consecutive_failures:
                print(f"\nSTOPPED after {consecutive} consecutive failures at frame {n}.")
                print(f"  last error: {failures[-1][1]}")
                print(f"  The volume has most likely dropped. Remount and re-run this "
                      f"exact command; completed files are skipped.")
                print(f"  staged so far: {copied} files, {total_bytes/1e9:.2f} GB")
                sys.exit(1)
            continue

        if i % 25 == 0:
            print(f"  {i}/{len(frames)} frames  ({copied} copied, {skipped} already present, "
                  f"{total_bytes/1e9:.2f} GB)")

    print(f"\ndone: {copied} files copied, {skipped} already present, "
          f"{total_bytes/1e9:.2f} GB")
    if failures:
        print(f"  {len(failures)} frames failed (not fatal, under the consecutive limit):")
        for n, msg in failures[:5]:
            print(f"    frame {n}: {msg}")

    have = paired_frames(dst_dir)
    print(f"  staged copy holds {len(have)} complete MAX/IRX pairs")
    if len(have) < 0.9 * len(frames):
        print(f"  WARNING: expected ~{len(frames)}. Re-run to fill the gaps before "
              f"analysing, or the sample will not span the flight as intended.")


if __name__ == "__main__":
    main()
