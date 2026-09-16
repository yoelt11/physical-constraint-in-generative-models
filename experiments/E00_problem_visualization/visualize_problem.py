"""Visualize the constrained four-bar linkage toy problem.

Three panels:
  (a) the mechanism itself at a few joint angles, showing the fixed base
      x1-x2 and the constrained links x2-x3, x3-x4, x4-x1;
  (b) the curve traced by the coupler point (midpoint of link x3-x4) as
      the joint angle sweeps [0, 2*pi), for both assembly branches. This
      is the honest picture of the nonconvex 1-D feasible manifold: x3
      and x4 themselves are *not* interesting to plot individually, since
      each constraint pins one of them to an exact circle (x3 to radius
      l23 around x2, x4 to radius l41 around x1 -- shown dashed) by
      construction. The nonconvexity only shows up in a quantity derived
      from both, such as this coupler point;
  (c) the designed target distribution over the intrinsic coordinates
      (joint angle theta, branch b) that a generative model must
      reproduce -- satisfying h(x) = 0 alone is not sufficient.
"""

import numpy as np
import matplotlib.pyplot as plt

# Link lengths (l12 is the fixed ground link, l23 the shortest link).
# Satisfies the Grashof condition with l23 adjacent to the ground link, so
# the joint angle theta can sweep the full circle (crank-rocker), and was
# chosen (by a small grid search) to give a visually non-degenerate coupler
# curve rather than one that nearly retraces itself.
L12, L23, L34, L41 = 1.0, 0.6, 1.3, 0.9

X1 = np.array([0.0, 0.0])
X2 = np.array([L12, 0.0])

BRANCH_COLORS = {+1: "#1f77b4", -1: "#d62728"}


def circle_intersections(c0, r0, c1, r1):
    """Intersections of two circles; returns (None, None) if none exist."""
    d = np.linalg.norm(c1 - c0)
    if d < 1e-9 or d > r0 + r1 or d < abs(r0 - r1):
        return None, None
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h_sq = r0 ** 2 - a ** 2
    if h_sq < 0:
        return None, None
    h = np.sqrt(max(h_sq, 0.0))
    mid = c0 + a * (c1 - c0) / d
    perp = np.array([-(c1 - c0)[1], (c1 - c0)[0]]) / d
    return mid + h * perp, mid - h * perp  # branch +1, branch -1


def solve_linkage(theta):
    """Return {+1: x4, -1: x4} for the given joint angle, skipping branches
    where the loop cannot close (near the crank-rocker's tangent limits)."""
    x3 = X2 + L23 * np.array([np.cos(theta), np.sin(theta)])
    x4_plus, x4_minus = circle_intersections(X1, L41, x3, L34)
    solutions = {}
    if x4_plus is not None:
        solutions[+1] = x4_plus
        solutions[-1] = x4_minus
    return x3, solutions


def sample_target_theta(rng, n):
    """Deliberately multimodal, imbalanced intrinsic target distribution."""
    modes = [(0.9, 0.20, 0.45), (3.3, 0.15, 0.35), (5.3, 0.25, 0.20)]
    means, stds, weights = zip(*modes)
    idx = rng.choice(len(modes), size=n, p=weights)
    theta = rng.normal(np.array(means)[idx], np.array(stds)[idx])
    branch = rng.choice([+1, -1], size=n, p=[0.7, 0.3])
    return np.mod(theta, 2 * np.pi), branch


def plot_mechanism(ax):
    snapshot_thetas = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_thetas)))
    for theta, color in zip(snapshot_thetas, colors):
        x3, solutions = solve_linkage(theta)
        if +1 not in solutions:
            continue
        x4 = solutions[+1]
        chain = np.array([X1, X2, x3, x4, X1])
        ax.plot(chain[:, 0], chain[:, 1], "-", color=color, lw=1.5,
                 label=rf"$\theta={np.degrees(theta):.0f}\degree$")
        ax.scatter(*x3, color=color, s=22, zorder=3)
        ax.scatter(*x4, color=color, s=22, zorder=3)
    for point, label in [(X1, "$x_1$"), (X2, "$x_2$")]:
        ax.scatter(*point, color="black", marker="s", s=45, zorder=4)
        ax.annotate(label, point, textcoords="offset points", xytext=(6, 6))
    ax.set_title("(a) Mechanism at 6 joint angles (branch $b=+1$)")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def plot_coupler_curve(ax):
    thetas = np.linspace(0, 2 * np.pi, 2000)
    curves = {+1: [], -1: []}
    for theta in thetas:
        x3, solutions = solve_linkage(theta)
        for branch, x4 in solutions.items():
            curves[branch].append(0.5 * (x3 + x4))  # coupler point

    circle_angles = np.linspace(0, 2 * np.pi, 200)
    x3_circle = X2 + L23 * np.c_[np.cos(circle_angles), np.sin(circle_angles)]
    x4_circle = X1 + L41 * np.c_[np.cos(circle_angles), np.sin(circle_angles)]
    ax.plot(*x3_circle.T, "--", color="gray", lw=0.8, alpha=0.6)
    ax.plot(*x4_circle.T, "--", color="gray", lw=0.8, alpha=0.6)

    for branch, points in curves.items():
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], color=BRANCH_COLORS[branch],
                 lw=1.5, label=f"branch $b={'+1' if branch == 1 else '-1'}$")
    ax.scatter(*X1, color="black", marker="s", s=45, zorder=4)
    ax.scatter(*X2, color="black", marker="s", s=45, zorder=4)
    ax.set_title("(b) Coupler-point curve (nonconvex feasible manifold)")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="upper right", fontsize=8)


def plot_target_distribution(ax):
    rng = np.random.default_rng(0)
    theta, branch = sample_target_theta(rng, 200_000)
    bins = np.linspace(0, 2 * np.pi, 121)
    for b in (+1, -1):
        ax.hist(theta[branch == b], bins=bins, color=BRANCH_COLORS[b], alpha=0.6,
                 label=f"branch $b={'+1' if b == 1 else '-1'}$")
    ax.set_title(r"(c) Target intrinsic distribution $p(\theta, b)$")
    ax.set_xlabel(r"joint angle $\theta$")
    ax.set_ylabel("count")
    ax.set_xlim(0, 2 * np.pi)
    ax.legend(loc="upper right", fontsize=8)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_mechanism(axes[0])
    plot_coupler_curve(axes[1])
    plot_target_distribution(axes[2])
    fig.suptitle("Constrained four-bar linkage: mechanism, feasible manifold, and target distribution")
    fig.tight_layout()

    out_dir = __file__.rsplit("/", 1)[0] + "/figures"
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir + "/problem_overview.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
