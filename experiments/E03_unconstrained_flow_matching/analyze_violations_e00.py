"""Which constraint is violated most, and where on the manifold do
violations concentrate? Breaks the aggregate |h(x)| residual (used in
plot_metrics_e00.py) into its three components, and looks at residual
as a function of position on the curve (theta, branch) rather than only
in aggregate -- to check the specific claim that certain regions of the
feasible manifold are harder for the unconstrained baseline to reach.

Usage:
    python3 analyze_violations_e00.py --n-samples 5000
"""

import argparse
import pathlib
import sys

import jax
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import flow_matching as fm  # noqa: E402
from e00_linkage_dataset import X1, X2, L23, L34, L41, solve_linkage_batch  # noqa: E402
from plot_metrics_e00 import infer_intrinsic  # noqa: E402


def per_constraint_residuals(x3, x4):
    h1 = np.linalg.norm(x3 - X2, axis=-1) - L23
    h2 = np.linalg.norm(x4 - x3, axis=-1) - L34
    h3 = np.linalg.norm(x4 - X1, axis=-1) - L41
    return {"h1 (x3-x2 length)": h1, "h2 (x4-x3 length)": h2, "h3 (x4-x1 length)": h3}


def plot_per_constraint(ax, components):
    labels = list(components.keys())
    data = [np.abs(components[k]) for k in labels]
    ax.boxplot(data, tick_labels=[l.split(" ")[0] for l in labels], showfliers=True,
               flierprops=dict(marker=".", markersize=3, alpha=0.3))
    ax.set_yscale("log")
    ax.set_ylabel(r"$|h_i(x)|$ (log scale)")
    ax.set_title("(a) Which constraint is violated most?", fontweight="bold")
    for i, k in enumerate(labels):
        print(f"  {k}: mean {np.mean(np.abs(components[k])):.4f}, "
              f"p99 {np.percentile(np.abs(components[k]), 99):.4f}, "
              f"max {np.max(np.abs(components[k])):.4f}")


def plot_residual_vs_theta(ax, theta, branch, resid):
    for b, color, marker in [(1, "#1f77b4", "o"), (-1, "#d62728", "x")]:
        mask = branch == b
        ax.scatter(theta[mask], resid[mask], s=8, alpha=0.4, color=color, marker=marker,
                   label=f"branch $b={'+1' if b == 1 else '-1'}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\theta$ (position on the manifold)")
    ax.set_ylabel(r"$|h(x)|$ (log scale)")
    ax.set_title("(b) Where do violations concentrate?", fontweight="bold")
    ax.legend(fontsize=8)


def plot_spatial(ax, coupler_pts, resid, coupler_plus, coupler_minus):
    ax.plot(*coupler_plus.T, color="black", lw=1, alpha=0.4)
    ax.plot(*coupler_minus.T, color="black", lw=1, alpha=0.4)
    order = np.argsort(resid)  # draw worst offenders on top
    sc = ax.scatter(coupler_pts[order, 0], coupler_pts[order, 1], c=np.log10(resid[order]),
                    cmap="RdYlGn_r", s=10, alpha=0.8, vmin=-3, vmax=-0.3)
    fig = ax.get_figure()
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$log_{10}|h(x)|$")
    ax.set_aspect("equal")
    ax.set_title("(c) Spatial location of violations", fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--checkpoint", default=REPO_ROOT / "checkpoints" / "e00_unconstrained.pkl")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    params = fm.load_params(args.checkpoint)
    rng = jax.random.PRNGKey(args.seed)
    final = np.asarray(fm.sample(rng, params, dim=4, n=args.n_samples, num_steps=100))
    x3, x4 = final[:, :2], final[:, 2:]

    components = per_constraint_residuals(x3, x4)
    resid = np.max(np.abs(np.stack(list(components.values()))), axis=0)
    theta_hat, branch_hat = infer_intrinsic(x3, x4)

    print(f"n = {args.n_samples}")
    print("per-constraint residual breakdown:")

    thetas = np.linspace(0, 2 * np.pi, 720)
    cx3, cx4_plus, cx4_minus = solve_linkage_batch(thetas)
    coupler_plus, coupler_minus = 0.5 * (cx3 + cx4_plus), 0.5 * (cx3 + cx4_minus)
    coupler_pts = 0.5 * (x3 + x4)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_per_constraint(axes[0], components)
    plot_residual_vs_theta(axes[1], theta_hat, branch_hat, resid)
    plot_spatial(axes[2], coupler_pts, resid, coupler_plus, coupler_minus)
    fig.suptitle(f"Toy Problem A, unconstrained baseline: where and how it violates $h(x)=0$ (n={args.n_samples})",
                 fontweight="bold")
    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "E03_e00_violation_analysis.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
