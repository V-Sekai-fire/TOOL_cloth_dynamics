#!/usr/bin/env python3
"""Time one full forward pass: PD vs AVBD, at matched fidelity.

The point of this harness is to avoid the comparison that produced the
"~880x faster" figure in Simulation.cpp:43. That number appears to have
been taken at AVBD_ITERS=1, which does not converge -- the AVBD/predictor
drift is identical at 1 and 16 iterations -- so it compares an
unconverged AVBD sweep against a converged PD solve. Iteration count is
precisely the thing being traded away, so it has to be held honest.

This runs the same demo three ways and reports both cost AND agreement:

    USE_PD=1          reference: DiffCloth's Eigen LLT + CG
    AVBD_ITERS=1      the cheap setting the old number came from
    AVBD_ITERS=16     what Simulation.cpp's own IFT comment says is
                      needed for the adjoint to be faithful

Timing hook: both modes print a single newline-terminated line
"forward started...0..20..…finished..." whose newline lands exactly when
the forward pass ends, so the delta from the preceding line is the cost
of one complete forward pass. We stop after the first pass -- the
optimizer's later iterations differ in count between configs and would
not be comparable.

Agreement is read from the per-step diagnostics that the AVBD path
prints: |Δx|_max for AVBD vs pd|Δx|_max for PD on the same step. Equal
displacement means the two solvers actually agree on the physics, which
is what makes a speed number meaningful.

Usage:
    python tests/bench_avbd_vs_pd.py [--demo sphere] [--exe build-fix/...]
"""

import argparse
import os
import re
import statistics
import subprocess
import sys
import time

STEPS = {"sphere": 350, "dress": 125, "tshirt": 250, "hat": 400, "sock": 400}

CONFIGS = [
    ("PD (Eigen LLT+CG)", {"USE_PD": "1"}),
    ("AVBD iters=1", {"AVBD_ITERS": "1"}),
    ("AVBD iters=16", {"AVBD_ITERS": "16"}),
]

RE_WALL = re.compile(r"wall=(\d+) us")
# The diagnostics embed a non-ASCII delta, so match around it rather than
# on it -- console encoding is not reliable here.
RE_AVBD_DX = re.compile(r"x\|_max=([0-9.eE+-]+)")
RE_PD_DX = re.compile(r"pd\|.?x\|_max=([0-9.eE+-]+)")


def run_one(exe, demo, seed, env_extra, timeout):
    """Run until the first forward pass completes; return timings."""
    env = dict(os.environ)
    env.update(env_extra)
    # Keep the solver honest about which device it picked.
    proc = subprocess.Popen(
        [exe, "-demo", demo, "-seed", str(seed)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        errors="replace",
        bufsize=1,
    )

    prev_t = None
    fwd_t0 = None
    forward_s = None
    walls = []
    avbd_dx = []
    pd_dx = []
    started = time.monotonic()

    try:
        for line in proc.stdout:
            now = time.monotonic()
            if "wall=" in line:
                m = RE_WALL.search(line)
                if m:
                    walls.append(int(m.group(1)))
            if "avbd-shadow" in line:
                m = RE_AVBD_DX.search(line)
                if m:
                    avbd_dx.append(float(m.group(1)))
            elif "pd-step" in line:
                m = RE_PD_DX.search(line)
                if m:
                    pd_dx.append(float(m.group(1)))
            # Timing hook. In PD mode the whole
            # "forward started...0..20..…340..finished..." arrives as ONE
            # line, flushed only when the pass ends. In AVBD mode the
            # per-step diagnostics interrupt it, so it closes after
            # "forward started...0..". Timing the "forward started" line
            # alone therefore measures the whole pass for PD and a single
            # step for AVBD -- a ~2400x artifact. So: start the clock at
            # the line BEFORE "forward started", and stop at whichever
            # comes first, the "finished..." that terminates the marker
            # run or the start of the backward pass.
            if fwd_t0 is None and "forward started" in line:
                fwd_t0 = prev_t if prev_t is not None else started
                if "finished" in line:          # PD: complete on one line
                    forward_s = now - fwd_t0
                    break
            elif fwd_t0 is not None and ("backward started" in line
                                         or "finished" in line):
                forward_s = now - fwd_t0
                break
            prev_t = now
            if now - started > timeout:
                break
    finally:
        proc.kill()
        proc.wait()

    return {
        "forward_s": forward_s,
        "walls": walls,
        "avbd_dx": avbd_dx,
        "pd_dx": pd_dx,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="sphere", choices=sorted(STEPS))
    ap.add_argument("--exe", default="build-fix/tool_cloth_dynamics.exe")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800)
    args = ap.parse_args()

    if not os.path.exists(args.exe):
        sys.exit("missing executable: %s" % args.exe)
    # CreateProcess wants a normalized absolute path; a forward-slash
    # relative one passes os.path.exists but fails to spawn on Windows.
    args.exe = os.path.normpath(os.path.abspath(args.exe))

    nsteps = STEPS[args.demo]
    print("demo=%s  steps/forward=%d  exe=%s\n" % (args.demo, nsteps, args.exe))

    rows = []
    for label, env_extra in CONFIGS:
        print("running %-20s %s" % (label, env_extra), flush=True)
        r = run_one(args.exe, args.demo, args.seed, env_extra, args.timeout)
        if r["forward_s"] is None:
            print("  (no forward pass completed within timeout)\n")
            rows.append((label, None, None, None))
            continue
        ms_step = r["forward_s"] * 1000.0 / nsteps
        solve_ms = statistics.mean(r["walls"]) / 1000.0 if r["walls"] else None
        # Agreement: AVBD displacement vs PD displacement on the same steps.
        ratio = None
        if r["avbd_dx"] and r["pd_dx"]:
            n = min(len(r["avbd_dx"]), len(r["pd_dx"]))
            a = statistics.median(r["avbd_dx"][:n])
            p = statistics.median(r["pd_dx"][:n])
            if p:
                ratio = a / p
        print("  forward pass %.2f s  ->  %.2f ms/step" % (r["forward_s"], ms_step))
        if solve_ms:
            print("  avbd solve dispatch mean %.2f ms/step" % solve_ms)
            # Sanity: the GPU dispatches are a subset of the forward
            # pass, so their total cannot exceed it. If it does, the
            # timing hook mis-measured (this caught a 2400x artifact).
            dispatch_total = sum(r["walls"]) / 1e6
            if dispatch_total > r["forward_s"] * 1.05:
                print("  !! IMPLAUSIBLE: solve dispatches total %.2f s > "
                      "forward pass %.2f s -- timing hook is wrong"
                      % (dispatch_total, r["forward_s"]))
        if ratio:
            print("  |dx| AVBD/PD = %.4f  (1.0 == same physics)" % ratio)
        print()
        rows.append((label, ms_step, solve_ms, ratio))

    print("=" * 66)
    print("%-22s %12s %12s %14s" % ("config", "ms/step", "solve ms", "|dx| vs PD"))
    print("-" * 66)
    base = next((r[1] for r in rows if r[0].startswith("PD") and r[1]), None)
    for label, ms, solve, ratio in rows:
        if ms is None:
            print("%-22s %12s" % (label, "n/a"))
            continue
        speed = ("%.1fx" % (base / ms)) if base else "-"
        print("%-22s %12.2f %12s %14s   %s" % (
            label, ms,
            ("%.2f" % solve) if solve else "-",
            ("%.4f" % ratio) if ratio else "-",
            speed))
    print("=" * 66)
    print("\nA speedup is only meaningful on a row whose |dx| vs PD is ~1.0.")


if __name__ == "__main__":
    main()
