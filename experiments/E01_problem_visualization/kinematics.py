"""Kinematics for the second toy problem: a two-legged biped stick figure.

One fixed hip x0, and two independent open kinematic chains (legs), each
with a knee and a foot -- i.e. two double pendulums sharing a pivot, like
E00's four-bar linkage but with a tree topology instead of a closed loop.
Each leg therefore has 2 free joint angles and no loop-closure constraint,
so (unlike E00) there is no discrete assembly branch: every joint angle
pair is reachable and valid.
"""

import pathlib

import numpy as np

X0 = np.array([0.0, 0.0])  # hip, fixed anchor

# Thigh/shin lengths, shared by both legs (a real biped has equal-length
# legs). Deliberately unequal within a leg (thigh != shin) so the foot's
# reachable region is a genuine annulus, not a degenerate full disk.
L1, L2 = 1.0, 0.8

LEG_COLORS = {"A": "#1f77b4", "B": "#d62728"}


def leg_kinematics(theta_hip, theta_knee):
    """Knee and foot position for one leg, given its two joint angles."""
    knee = X0 + L1 * np.array([np.cos(theta_hip), np.sin(theta_hip)])
    foot = knee + L2 * np.array([np.cos(theta_knee), np.sin(theta_knee)])
    return knee, foot


def save_figure(fig, description, dpi=200, **savefig_kwargs):
    """Save into the repo-root figures/ folder as "<experiment-id>_<description>.png"."""
    experiment_dir = pathlib.Path(__file__).resolve().parent
    experiment_id = experiment_dir.name.split("_", 1)[0]
    figures_dir = experiment_dir.parent.parent / "figures"
    out_path = figures_dir / f"{experiment_id}_{description}.png"
    fig.savefig(out_path, dpi=dpi, **savefig_kwargs)
    print(f"Saved {out_path}")
