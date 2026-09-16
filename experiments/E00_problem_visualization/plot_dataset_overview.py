"""Dataset overview: the mechanism (left) and the reachable loci of the
free nodes x3, x4 (right), for presenting the dataset alongside the
problem statement.

Note what the right panel is and is not showing: x3 and x4 are each
pinned to an *exact* circle by a single constraint (x3 to radius l23
around x2, x4 to radius l41 around x1), so individually their loci are
not interesting -- this panel is about which *arc* of each circle is
reachable, and in particular the asymmetry between the two assembly
branches for x4. The true nonconvex feasible manifold (which needs both
x3 and x4 jointly) is shown separately in the coupler-point curve of
visualize_problem.py.
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from visualize_problem import X1, X2, BRANCH_COLORS, solve_linkage, save_figure

SCHEMATIC_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "figures" / "E00_mechanism_schematic.png"
)


def plot_schematic(ax):
    """Embed the pre-rendered single-pose schematic; return its aspect
    ratio (height / width) so the other panel can match its box shape."""
    img = mpimg.imread(SCHEMATIC_PATH)
    ax.imshow(img)
    # Keep the box border visible (matching panel b's frame) instead of
    # ax.axis("off") -- the two axes boxes are the same size regardless,
    # but an invisible border made panel (a) look shorter, since most of
    # the embedded image is its own title/legend whitespace, not ink.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(a) Mechanism (single pose, labeled)", fontweight="bold")
    return img.shape[0] / img.shape[1]


def plot_node_loci(ax, box_aspect):
    thetas = np.linspace(0, 2 * np.pi, 2000)
    x3_locus = []
    x4_locus = {+1: [], -1: []}
    for theta in thetas:
        x3, solutions = solve_linkage(theta)
        x3_locus.append(x3)
        for branch, x4 in solutions.items():
            x4_locus[branch].append(x4)
    x3_locus = np.array(x3_locus)

    ax.plot(*x3_locus.T, color="#555555", lw=1.5, label=r"$x_3$ locus (radius $\ell_{23}$ around $x_2$)")
    for branch, pts in x4_locus.items():
        pts = np.array(pts)
        tag = "+1" if branch == 1 else "-1"
        ax.plot(*pts.T, color=BRANCH_COLORS[branch], lw=1.5,
                 label=rf"$x_4$ locus, branch $b={tag}$")

    ax.scatter(*X1, color="black", marker="s", s=45, zorder=4)
    ax.scatter(*X2, color="black", marker="s", s=45, zorder=4)
    ax.annotate("$x_1$", X1, textcoords="offset points", xytext=(6, 6))
    ax.annotate("$x_2$", X2, textcoords="offset points", xytext=(6, 6))
    ax.set_title("(b) Reachable positions of $x_3$, $x_4$", fontweight="bold")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    # Match the schematic's box shape (not just the data aspect) so both
    # panels render at the same height.
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(box_aspect)
    ax.legend(loc="upper right", fontsize=7)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    box_aspect = plot_schematic(axes[0])
    plot_node_loci(axes[1], box_aspect)
    fig.suptitle("One fixed four-body mechanism: many valid poses", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "dataset_overview", bbox_inches="tight")


if __name__ == "__main__":
    main()
