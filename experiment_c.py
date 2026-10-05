"""
Experiment C: analyse sampling intervals of a Physics Toolbox gyroscope CSV.

Usage:
    python experiment_c.py trial1.csv
    python experiment_c.py trial1.csv trial2.csv trial3.csv
    python experiment_c.py trial1.csv --axis z
The original CSV files are only read, never modified.

How problem rows are handled (reported in the output):
  * Invalid rows (wrong number of columns, text instead of numbers, NaN/inf)
    are skipped and counted.
  * Timestamps that do not increase (duplicate: dt = 0, or going backwards:
    dt < 0) are removed, keeping the first sample at that time. Dt statistics
    and integration use only this cleaned, strictly increasing series.
"""
import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt

STILL_SECONDS = 2.0
RAD2DEG = 180.0 / math.pi
AXES = {"x": 1, "y": 2, "z": 3}      # column index after the time column


def read_csv(path):
    """Parse the file; return time, omega (rad/s per axis) and row counts."""
    rows, invalid = [], []
    n_data_lines = 0
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("time"):
                continue                          # blank, metadata or header
            n_data_lines += 1
            text = line.replace("\u2212", "-").replace(",", ".")
            try:
                parts = text.split(";")
                if len(parts) < 4:
                    raise ValueError("fewer than 4 columns")
                vals = [float(v) for v in parts[:4]]   # time, x, y, z
                if not all(math.isfinite(v) for v in vals):
                    raise ValueError("NaN or infinite value")
                rows.append(vals)
            except ValueError as err:
                invalid.append((lineno, str(err)))
    data = np.array(rows)
    t = data[:, 0]
    omega = {a: data[:, i] for a, i in AXES.items()}
    return t, omega, n_data_lines, invalid


def drop_nonincreasing(t, omega):
    """Keep only samples whose timestamp is strictly later than the last kept one."""
    keep = [0]
    for i in range(1, len(t)):
        if t[i] > t[keep[-1]]:
            keep.append(i)
    keep = np.array(keep)
    return t[keep], {a: w[keep] for a, w in omega.items()}


def trapezoid_cumulative(w, t):
    return np.concatenate(([0.0], np.cumsum(0.5 * (w[:-1] + w[1:]) * np.diff(t))))


def analyse(path, forced_axis=None):
    name = os.path.splitext(os.path.basename(path))[0]
    t_raw, omega_raw, n_lines, invalid = read_csv(path)

    # ---- timestamp quality on the valid rows, before cleaning ----
    dt_raw = np.diff(t_raw)
    n_dup = int(np.sum(dt_raw == 0))
    n_back = int(np.sum(dt_raw < 0))
    t, omega_rad = drop_nonincreasing(t_raw, omega_raw)
    n_removed = len(t_raw) - len(t)

    # ---- sampling period statistics (milliseconds) ----
    dt_ms = np.diff(t) * 1000.0
    mean_ms = dt_ms.mean()
    std_ms = dt_ms.std(ddof=1) if len(dt_ms) > 1 else float("nan")   # n-1 denominator
    min_ms, max_ms = dt_ms.min(), dt_ms.max()

    print(f"\n===== {os.path.basename(path)} =====")
    print("--- Data handling ---")
    print(f"Data rows in file            : {n_lines}")
    print(f"Invalid rows skipped         : {len(invalid)}")
    for lineno, why in invalid[:10]:
        print(f"    line {lineno}: {why}")
    if len(invalid) > 10:
        print(f"    ... and {len(invalid) - 10} more")
    print(f"Valid rows                   : {len(t_raw)}")
    print(f"Duplicate timestamps (dt = 0): {n_dup}")
    print(f"Backwards timestamps (dt < 0): {n_back}")
    print(f"Non-increasing pairs (total) : {n_dup + n_back}")
    print(f"Rows removed for this reason : {n_removed}")
    print(f"Samples used (number of samples): {len(t)}")

    print("\n--- Sampling period (ms) ---")
    print(f"Intervals                    : {len(dt_ms)}")
    print(f"Average period               : {mean_ms:.4f} ms")
    print(f"Sample std (n-1)             : {std_ms:.4f} ms")
    print(f"Minimum period               : {min_ms:.4f} ms")
    print(f"Maximum period               : {max_ms:.4f} ms")
    print(f"Effective mean rate          : {1000.0 / mean_ms:.2f} Hz")
    print(f"Recording duration           : {t[-1] - t[0]:.3f} s")

    # ---- angle estimate (same method as Experiment B) ----
    omega_deg = {a: w * RAD2DEG for a, w in omega_rad.items()}
    still = t <= t[0] + STILL_SECONDS
    res = {}
    for a, w in omega_deg.items():
        b = w[still].mean()
        theta = trapezoid_cumulative(w - b, t)
        res[a] = (b, w - b, theta)
    axis = forced_axis or max(res, key=lambda a: abs(res[a][2][-1]))
    b, wc, theta = res[axis]
    print(f"\n--- Angle (axis {axis.upper()}"
          f"{', forced' if forced_axis else ', largest |angle|'}) ---")
    print(f"Bias b                       : {b:.4f} deg/s")
    print(f"Signed theta*                : {theta[-1]:.2f} deg")

    # ---- plot 1: angular velocity vs time ----
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, omega_deg[axis], alpha=0.5, label="raw")
    ax.plot(t, wc, label="bias-corrected")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Angular velocity ω{axis} (deg/s)")
    ax.set_title(f"{name}: angular velocity")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{name}_C_angular_velocity.png", dpi=150)
    plt.close(fig)

    # ---- plot 2: cumulative angle vs time ----
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, theta)
    ax.axhline(math.copysign(90, theta[-1]), ls="--", c="r", label="±90° reference")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative angle θ* (deg)")
    ax.set_title(f"{name}: cumulative angle")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{name}_C_cumulative_angle.png", dpi=150)
    plt.close(fig)

    # ---- extra: histogram of sampling periods (optional for the report) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dt_ms, bins=50)
    ax.set_xlabel("Sampling period Δt (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"{name}: sampling periods")
    fig.tight_layout()
    fig.savefig(f"{name}_C_dt_histogram.png", dpi=150)
    plt.close(fig)
    print(f"Plots saved: {name}_C_angular_velocity.png, "
          f"{name}_C_cumulative_angle.png, {name}_C_dt_histogram.png")

    return dict(file=os.path.basename(path), n=len(t), invalid=len(invalid),
                nonincr=n_dup + n_back, mean=mean_ms, std=std_ms,
                mn=min_ms, mx=max_ms)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--axis", choices=["x", "y", "z"])
    args = p.parse_args()
    out = [analyse(f, args.axis) for f in args.files]
    if len(out) > 1:
        print("\n===== Summary (periods in ms) =====")
        print(f"{'file':<38}{'N':>6}{'invalid':>8}{'non-incr':>9}"
              f"{'mean':>9}{'std':>9}{'min':>9}{'max':>9}")
        for r in out:
            print(f"{r['file']:<38}{r['n']:>6}{r['invalid']:>8}{r['nonincr']:>9}"
                  f"{r['mean']:>9.4f}{r['std']:>9.4f}{r['mn']:>9.4f}{r['mx']:>9.4f}")
