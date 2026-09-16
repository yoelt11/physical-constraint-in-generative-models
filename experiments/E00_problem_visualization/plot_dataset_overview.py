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

import numpy as np
import matplotlib.pyplot as plt

from visualize_problem import X1, X2, BRANCH_COLORS, solve_linkage, save_figure, plot_mechanism


def plot_mechanism_snapshots(ax):
    plot_mechanism(ax)
    ax.set_title("(a) One mechanism, several poses")


def plot_node_loci(ax):
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
                 label=rf"$x_4$ locus, branch $b={tag}$ (radius $\ell_{{41}}$ around $x_1$)")

    ax.scatter(*X1, color="black", marker="s", s=45, zorder=4)
    ax.scatter(*X2, color="black", marker="s", s=45, zorder=4)
    ax.annotate("$x_1$", X1, textcoords="offset points", xytext=(6, 6))
    ax.annotate("$x_2$", X2, textcoords="offset points", xytext=(6, 6))
    ax.set_title("(b) Reachable positions of $x_3$, $x_4$ across all poses")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    plot_mechanism_snapshots(axes[0])
    plot_node_loci(axes[1])
    fig.suptitle("One fixed four-body mechanism: many valid poses")
    fig.tight_layout()
    save_figure(fig, "dataset_overview", bbox_inches="tight")


if __name__ == "__main__":
    main()
