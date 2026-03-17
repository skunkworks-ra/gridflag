#!/usr/bin/env python
"""
Profiling harness for gridflag.

Modes:
  --debug      Enable DEBUG logging (per-step timings, Numba thread count).
               This is the cheapest diagnostic — run first.
  --cprofile   Python call-graph profiling via cProfile (adds ~5-10% overhead).
  --lineprof   Per-line timing on hot functions via line_profiler (adds ~10-30x).

For a py-spy flame graph (captures Numba threads, unlike cProfile):
    py-spy record -o flame.svg --native -- \\
        python profile_run.py --workers N

Usage:
    python profile_run.py [--debug] [--cprofile] [--lineprof] [--workers N] MS_PATH
"""
from __future__ import annotations

import argparse
import cProfile
import logging
import pstats
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "profile_results"
RESULTS_DIR.mkdir(exist_ok=True)


def _call_gridflag(ms_path: str, n_workers: int, device: str = "auto"):
    from gridflag.config import GridFlagConfig
    from gridflag.pipeline import run
    cfg = GridFlagConfig(n_workers=n_workers)
    return run(ms_path, config=cfg, device=device)


def run_with_cprofile(ms_path: str, n_workers: int, device: str = "auto"):
    pr = cProfile.Profile()
    pr.enable()
    _call_gridflag(ms_path, n_workers, device=device)
    pr.disable()

    out_path = RESULTS_DIR / "cprofile.txt"
    with open(out_path, "w") as f:
        ps = pstats.Stats(pr, stream=f)
        ps.sort_stats("cumulative")
        ps.print_stats(60)

    ps2 = pstats.Stats(pr)
    ps2.sort_stats("tottime")
    ps2.print_stats(40)
    print(f"\n[cProfile] Full stats written to {out_path}")


def run_with_line_profiler(ms_path: str, n_workers: int, device: str = "auto"):
    from line_profiler import LineProfiler

    import gridflag.histogram as hist_mod
    import gridflag.pipeline as pipe_mod

    lp = LineProfiler()
    lp.add_function(hist_mod._pass0_read_chunk)
    lp.add_function(hist_mod.parallel_histogram_fill)
    lp.add_function(hist_mod._extract_chunk)
    lp.add_function(hist_mod.compute_cell_stats_streaming)
    lp.add_function(pipe_mod.run)

    from gridflag.config import GridFlagConfig
    from gridflag.pipeline import run
    cfg = GridFlagConfig(n_workers=n_workers)
    lp(run)(ms_path, config=cfg, device=device)

    out_path = RESULTS_DIR / "line_profile.txt"
    with open(out_path, "w") as f:
        lp.print_stats(stream=f, output_unit=1e-3)  # ms
    lp.print_stats(output_unit=1e-3)
    print(f"\n[line_profiler] Written to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ms_path", help="Path to CASA Measurement Set")
    ap.add_argument("--debug", action="store_true",
                    help="Enable DEBUG logging (per-step timings, Numba thread count)")
    ap.add_argument("--cprofile", action="store_true", help="Run with cProfile")
    ap.add_argument("--lineprof", action="store_true", help="Run with line_profiler")
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of workers (0 = auto-detect)")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                    help="Compute device (default: auto)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    )
    if args.debug:
        # Only elevate gridflag loggers, not every third-party library.
        logging.getLogger("gridflag").setLevel(logging.DEBUG)

    if not args.cprofile and not args.lineprof:
        # Default: plain run (debug logging already enabled above if --debug).
        _call_gridflag(args.ms_path, args.workers, device=args.device)
    else:
        if args.cprofile:
            run_with_cprofile(args.ms_path, args.workers, device=args.device)
        if args.lineprof:
            run_with_line_profiler(args.ms_path, args.workers, device=args.device)


if __name__ == "__main__":
    main()
