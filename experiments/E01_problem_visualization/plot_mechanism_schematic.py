"""Labeled schematic of the two-legged biped at a single pose.

The anchor x0 (hip) is drawn distinctly from the four free nodes
(the two knees and two feet), with every link labeled by its length
symbol, mirroring E00's mechanism_schematic.png.
"""

import numpy as np
import matplotlib.pyplot as plt

from kinematics import X0, L1, L2, LEG_COLORS, leg_kinematics, save_figure

ANCHOR_COLOR = "#2b2b2b"

# One representative pose: hip angle, knee angle (degrees) per leg.
POSE = {"A": (-60.0, -90.0), "B": (-120.0, -90.0)}
LABEL_OFFSET = {"A": (10, -14), "B": (-70, -14)}


def midpoint(a, b):
    return 0.5 * (a + b)


def main():
    fig, ax = plt.subplots(figsize=(6, 6))

    for leg, (hip_deg, knee_deg) in POSE.items():
        theta_hip, theta_knee = np.radians(hip_deg), np.radians(knee_deg)
        knee, foot = leg_kinematics(theta_hip, theta_knee)
        color = LEG_COLORS[leg]
        knee_idx, foot_idx = ("1", "2") if leg == "A" else ("3", "4")

        for a, b, label in [(X0, knee, rf"$\ell_{{1{leg}}}$"),
                             (knee, foot, rf"$\ell_{{2{leg}}}$")]:
            ax.plot([a[0], b[0]], [a[1], b[1]], "-", color=color, lw=2.5, zorder=1)
            m = midpoint(a, b)
            ax.annotate(label, m, textcoords="offset points", xytext=(6, 4),
                         fontsize=11, color=color)

        ax.scatter(*knee, color=color, marker="o", s=140, zorder=3)
        ax.scatter(*foot, color=color, marker="o", s=140, zorder=3)
        ax.annotate(f"$x_{{{knee_idx}}}$ (knee {leg})", knee,
                     textcoords="offset points", xytext=(8, 8), fontsize=11)
        ax.annotate(f"$x_{{{foot_idx}}}$ (foot {leg})", foot,
                     textcoords="offset points", xytext=LABEL_OFFSET[leg], fontsize=11)

    ax.scatter(*X0, color=ANCHOR_COLOR, marker="s", s=160, zorder=4,
               label="anchor (fixed)")
    ax.annotate("$x_0$ (hip)", X0, textcoords="offset points", xytext=(10, 6), fontsize=12)
    ax.scatter([], [], color=LEG_COLORS["A"], marker="o", s=140, label="leg A (free)")
    ax.scatter([], [], color=LEG_COLORS["B"], marker="o", s=140, label="leg B (free)")

    ax.set_title("Two-legged biped: fixed hip $x_0$, two independent leg chains")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9)
    fig.tight_layout()
    save_figure(fig, "mechanism_schematic", bbox_inches="tight")


if __name__ == "__main__":
    main()
