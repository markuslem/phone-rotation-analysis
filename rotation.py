"""
Experiment B: estimate a ~90 degree rotation by integrating gyroscope data.

Usage:
    python experiment_b.py trial1.csv trial2.csv trial3.csv
    python experiment_b.py trial1.csv --axis z        # force an axis after inspecting
The original CSV files are only read, never modified.
"""

import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt

REFERENCE_DEG = 90.0
STILL_SECONDS = 2.0  # first still interval required by the experiment
RAD2DEG = 180.0 / math.pi  # rad/s -> deg/s
AXES = {"x": 1, "y": 2, "z": 3}  # column index after time


def read_csv(path):
    """Read Physics Toolbox CSV: '#' metadata, ';' separator, ',' decimals, '−' minus."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("time"):  # header row
                continue
            line = line.replace("\u2212", "-").replace(",", ".")
            rows.append([float(v) for v in line.split(";")])
    data = np.array(rows)
    t = data[:, 0]  # seconds
    omega = {a: data[:, i] for a, i in AXES.items()}  # rad/s
    return t, omega


def trapezoid_cumulative(omega, t):
    """Cumulative trapezoidal integral: sum of (w_i + w_i+1)/2 * (t_i+1 - t_i)."""
    dt = np.diff(t)  # real timestamp gaps
    steps = 0.5 * (omega[:-1] + omega[1:]) * dt
    return np.concatenate(([0.0], np.cumsum(steps)))


def analyse(path, forced_axis=None):
    t, omega_rad = read_csv(path)
    omega_deg = {a: w * RAD2DEG for a, w in omega_rad.items()}  # deg/s

    still = t <= t[0] + STILL_SECONDS
    print(f"\n===== {os.path.basename(path)} =====")
    print(
        f"Samples: {len(t)}, duration: {t[-1] - t[0]:.3f} s, "
        f"mean sampling rate: {(len(t) - 1) / (t[-1] - t[0]):.1f} Hz"
    )
    print(f"Still window for bias: first {STILL_SECONDS} s ({still.sum()} samples)")

    # --- axis inspection: bias-correct every axis and see which one rotated ---
    print("\nAxis evidence (bias from still window, then integrated):")
    print(f"{'axis':>4} {'bias deg/s':>11} {'peak |w*| deg/s':>16} {'theta* deg':>11}")
    results = {}
    for a, w in omega_deg.items():
        b = w[still].mean()
        wc = w - b
        theta = trapezoid_cumulative(wc, t)
        results[a] = (b, wc, theta)
        print(f"{a:>4} {b:11.4f} {np.max(np.abs(wc)):16.2f} {theta[-1]:11.2f}")

    axis = forced_axis or max(results, key=lambda a: abs(results[a][2][-1]))
    print(
        f"\nSelected axis: {axis.upper()} "
        f"({'user-forced' if forced_axis else 'largest |integrated angle|'}) "
        f"- check this against how you physically held the phone."
    )

    b_deg, wc, theta = results[axis]
    b_rad = omega_rad[axis][still].mean()
    theta_final = theta[-1]
    err = 100.0 * abs(abs(theta_final) - REFERENCE_DEG) / REFERENCE_DEG

    # warn if the phone seems to move inside the "still" window
    if np.max(np.abs(wc[still])) > 3.0:
        print(
            "WARNING: motion > 3 deg/s inside the first 2 s; the still interval "
            "may be contaminated. Check the plot."
        )

    print("\n--- Report values ---")
    print(f"Axis                : {axis.upper()}")
    print(f"Original unit       : rad/s")
    print(f"Bias b              : {b_rad:.6f} rad/s = {b_deg:.4f} deg/s")
    print(f"Signed theta*       : {theta_final:.2f} deg")
    print(f"|theta*|            : {abs(theta_final):.2f} deg")
    print(f"Reference           : {REFERENCE_DEG:.0f} deg")
    print(f"Percentage error    : {err:.2f} %")

    # --- plots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(t, omega_deg[axis], label="raw", alpha=0.5)
    ax1.plot(t, wc, label="bias-corrected")
    ax1.axvspan(
        t[0], t[0] + STILL_SECONDS, color="grey", alpha=0.15, label="bias window"
    )
    ax1.set_ylabel(f"ω{axis} (deg/s)")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(t, theta)
    ax2.axhline(
        math.copysign(REFERENCE_DEG, theta_final),
        ls="--",
        c="r",
        label=f"±{REFERENCE_DEG:.0f}° reference",
    )
    ax2.set_ylabel("Cumulative angle (deg)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True)
    fig.suptitle(f"{os.path.basename(path)} - axis {axis.upper()}")
    fig.tight_layout()
    out = os.path.splitext(os.path.basename(path))[0] + "_plot.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out}")
    return theta_final, err


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--axis", choices=["x", "y", "z"])
    args = p.parse_args()
    summary = [(f, *analyse(f, args.axis)) for f in args.files]
    print("\n===== Summary =====")
    for f, th, e in summary:
        print(f"{os.path.basename(f)}: theta* = {th:.2f} deg, error = {e:.2f} %")
