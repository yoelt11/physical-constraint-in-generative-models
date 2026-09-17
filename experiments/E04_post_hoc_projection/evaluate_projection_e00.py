"""Post-hoc projection onto h(x) = 0 for the E03 unconstrained baseline,
via Gauss-Newton (src/projection.py). Pure post-processing -- the model
itself is unchanged, only its output is corrected -- so this checks two
things: does projection fix the constraint-error axis (it should, by
construction), and does it preserve the distribution-coverage axis
already measured to be excellent for the unconstrained baseline, or
does fixing constraint violation quietly distort it -- especially for
branch b=-1, already found to be the harder region
(analyze_violations_e00.py).

Usage:
    python3 evaluate_projection_e00.py --n-samples 5000
"""

import argparse
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import flow_matching as fm  # noqa: E402
from projection import gauss_newton_project  # noqa: E402
from metrics import wasserstein_1d, binary_js_divergence  # noqa: E402
from e00_linkage_dataset import (  # noqa: E402
    X1, X2, L23, L34, L41, BRANCH_PROB_PLUS, sample_theta_and_branch,
    constraint_residuals, infer_intrinsic,
)


def h_e00(x):
    """The constraint function from main.typ (squared-distance form), for
    a single point x = concat(x3, x4) in R^4. JAX-differentiable -- its
    Jacobian is obtained automatically by src/projection.py."""
    x3, x4 = x[:2], x[2:]
    h1 = jnp.sum((x3 - X2) ** 2) - L23 ** 2
    h2 = jnp.sum((x4 - x3) ** 2) - L34 ** 2
    h3 = jnp.sum((x4 - X1) ** 2) - L41 ** 2
    return jnp.array([h1, h2, h3])


def plot_constraint_error(ax, resid_before, resid_after):
    bins = np.logspace(np.log10(max(np.min(resid_after[resid_after > 0]), 1e-12)),
                        np.log10(resid_before.max()), 40)
    ax.hist(resid_before, bins=bins, color="#d62728", alpha=0.6, label="before (unconstrained)")
    ax.hist(np.clip(resid_after, bins[0], None), bins=bins, color="#2ca02c", alpha=0.6,
             label="after (projected)")
    ax.set_xscale("log")
    ax.set_title("(a) Constraint error, before vs. after", fontweight="bold")
    ax.set_xlabel(r"$|h(x)|$ (log scale)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)


def plot_theta_coverage(ax, theta_true, theta_before, theta_after, w1_before, w1_after):
    bins = np.linspace(0, 2 * np.pi, 61)
    ax.hist(theta_true, bins=bins, color="#888888", alpha=0.5, label="true target")
    ax.hist(theta_before, bins=bins, histtype="step", color="#d62728", lw=1.5,
             label=rf"before ($W_1$={w1_before:.3f})")
    ax.hist(theta_after, bins=bins, histtype="step", color="#2ca02c", lw=1.5,
             label=rf"after ($W_1$={w1_after:.3f})")
    ax.set_title("(b) Joint-angle coverage, before vs. after", fontweight="bold")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("count")
    ax.set_xlim(0, 2 * np.pi)
    ax.legend(fontsize=8)


def plot_branch_coverage(ax, p_true, p_before, p_after, js_before, js_after):
    x = np.arange(2)
    width = 0.25
    ax.bar(x - width, [p_true, 1 - p_true], width, color="#888888", alpha=0.6, label="true target")
    ax.bar(x, [p_before, 1 - p_before], width, color="#d62728", alpha=0.6,
           label=f"before (JS={js_before:.4f})")
    ax.bar(x + width, [p_after, 1 - p_after], width, color="#2ca02c", alpha=0.6,
           label=f"after (JS={js_after:.4f})")
    ax.set_xticks(x, ["$b=+1$", "$b=-1$"])
    ax.set_title("(c) Branch coverage, before vs. after", fontweight="bold")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8)


def plot_branch_residual(ax, branch_before, resid_before, branch_after, resid_after):
    """Does projection equally fix both branches, or does the harder
    branch (b=-1, per analyze_violations_e00.py) stay harder?"""
    data = [resid_before[branch_before == 1], resid_after[branch_after == 1],
            resid_before[branch_before == -1], resid_after[branch_after == -1]]
    ax.boxplot(data, tick_labels=["b=+1\nbefore", "b=+1\nafter", "b=-1\nbefore", "b=-1\nafter"],
               showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.3))
    ax.set_yscale("log")
    ax.set_ylabel(r"$|h(x)|$ (log scale)")
    ax.set_title("(d) Does projection fix both branches equally?", fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--checkpoint", default=REPO_ROOT / "checkpoints" / "e00_unconstrained.pkl")
    parser.add_argument("--seed", type=int, default=2)  # same seed as plot_metrics_e00.py
    args = parser.parse_args()

    params = fm.load_params(args.checkpoint)
    rng = jax.random.PRNGKey(args.seed)
    before = np.asarray(fm.sample(rng, params, dim=4, n=args.n_samples, num_steps=100))
    after = np.asarray(gauss_newton_project(jnp.asarray(before), h_e00))

    x3_b, x4_b = before[:, :2], before[:, 2:]
    x3_a, x4_a = after[:, :2], after[:, 2:]

    resid_before = constraint_residuals(x3_b, x4_b)
    resid_after = constraint_residuals(x3_a, x4_a)
    theta_before, branch_before = infer_intrinsic(x3_b, x4_b)
    theta_after, branch_after = infer_intrinsic(x3_a, x4_a)

    true_theta, _ = sample_theta_and_branch(np.random.default_rng(args.seed), args.n_samples)
    w1_before = wasserstein_1d(theta_before, true_theta)
    w1_after = wasserstein_1d(theta_after, true_theta)
    p_before = float(np.mean(branch_before == 1))
    p_after = float(np.mean(branch_after == 1))
    js_before = binary_js_divergence(p_before, BRANCH_PROB_PLUS)
    js_after = binary_js_divergence(p_after, BRANCH_PROB_PLUS)

    print(f"n = {args.n_samples}")
    print("constraint error:")
    print(f"  before: mean {resid_before.mean():.4f}, median {np.median(resid_before):.4f}, "
          f"p99 {np.percentile(resid_before, 99):.4f}, max {resid_before.max():.4f}")
    print(f"  after:  mean {resid_after.mean():.2e}, median {np.median(resid_after):.2e}, "
          f"p99 {np.percentile(resid_after, 99):.2e}, max {resid_after.max():.2e}")
    print("theta coverage (W1, rad):")
    print(f"  before: {w1_before:.4f}   after: {w1_after:.4f}")
    print("branch coverage (P(b=+1), JS):")
    print(f"  before: {p_before:.3f}, JS={js_before:.4f}   after: {p_after:.3f}, JS={js_after:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_constraint_error(axes[0, 0], resid_before, resid_after)
    plot_theta_coverage(axes[0, 1], true_theta, theta_before, theta_after, w1_before, w1_after)
    plot_branch_coverage(axes[1, 0], BRANCH_PROB_PLUS, p_before, p_after, js_before, js_after)
    plot_branch_residual(axes[1, 1], branch_before, resid_before, branch_after, resid_after)
    fig.suptitle("Post-hoc projection, Toy Problem A: does it fix error without breaking coverage? "
                  f"(n={args.n_samples})", fontweight="bold")
    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "E04_e00_projection_before_after.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
