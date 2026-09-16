"""Dataset overview for the two-legged biped: the mechanism (left) and the
reachable region of each foot (right), mirroring E00's Figure 1.

Unlike E00 -- where each free node was pinned to an exact 1-D circle by a
single constraint -- each leg here has 2 independent joint angles and no
loop-closure constraint, so a foot's reachable set is a genuine 2-D
annulus (inner radius |l1-l2|, outer radius l1+l2), not a curve. That is
the direct geometric consequence of the tree topology having 2 DOF per
leg, versus E00's closed loop having 1 DOF in total.
"""

import pathlib

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle, Wedge

from kinematics import X0, L1, L2, save_figure

SCHEMATIC_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "figures" / "E01_mechanism_schematic.png"
)


def plot_schematic(ax):
    img = mpimg.imread(SCHEMATIC_PATH)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) Mechanism (single pose, labeled)", fontweight="bold")
    return img.shape[0] / img.shape[1]


def plot_reachable_region(ax, box_aspect):
    knee_radius = L1
    outer_radius = L1 + L2
    inner_radius = abs(L1 - L2)

    ax.add_patch(Wedge(X0, outer_radius, 0, 360, width=outer_radius - inner_radius,
                        facecolor="#d62728", alpha=0.25, edgecolor="none",
                        label=r"reachable region of either foot"))
    ax.add_patch(Circle(X0, knee_radius, fill=False, linestyle="--",
                         color="gray", lw=1.0, label=r"reachable circle of either knee"))
    ax.scatter(*X0, color="black", marker="s", s=45, zorder=4)
    ax.annotate("$x_0$", X0, textcoords="offset points", xytext=(6, 6))

    lim = outer_radius * 1.1
    ax.plot([-lim, lim], [-lim, lim], alpha=0)  # invisible: sets data limits without a warning
    ax.set_title("(b) Reachable region of a knee, a foot", fontweight="bold")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(box_aspect)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 5.2))
    box_aspect = plot_schematic(axes[0])
    plot_reachable_region(axes[1], box_aspect)
    fig.suptitle("One fixed hip, two independent legs: a tree, not a loop", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "dataset_overview", bbox_inches="tight")


if __name__ == "__main__":
    main()
