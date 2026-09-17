"""Companion metrics figure for E03_e00_unconstrained_flow.gif: the two
independent evaluation axes from main.typ's Evaluation section --
constraint error and distributional coverage -- computed over a large
sample (the animation's 600 points are for visual clarity, not
statistics).

(a) Constraint error: distribution of |h(x)| over many samples.
(b) Distribution coverage: generated vs. true theta, recovered from the
    model's own (x3, x4) output by inverting the forward kinematics --
    theta_hat from the angle of x3, branch_hat from which side of the
    circle-intersection line x4 falls on -- so this is a property of the
    model's raw output, not something read off a label.
(c) Branch coverage: generated vs. true P(b = +1).

Usage:
    python3 plot_metrics_e00.py --n-samples 5000
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
from metrics import wasserstein_1d, binary_js_divergence  # noqa: E402
from e00_linkage_dataset import (  # noqa: E402
    BRANCH_PROB_PLUS, sample_theta_and_branch, constraint_residuals, infer_intrinsic,
)


def plot_constraint_error(ax, resid):
    bins = np.logspace(np.log10(max(resid.min(), 1e-4)), np.log10(resid.max()), 40)
    ax.hist(resid, bins=bins, color="#d62728", alpha=0.75)
    ax.set_xscale("log")
    for p, style in [(50, "-"), (90, "--"), (99, ":")]:
        val = np.percentile(resid, p)
        ax.axvline(val, color="black", lw=1, linestyle=style, label=f"p{p} = {val:.3f}")
    ax.set_title("(a) Constraint error $|h(x)|$", fontweight="bold")
    ax.set_xlabel(r"$|h(x)|$ (log scale)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)


def plot_theta_coverage(ax, theta_gen, theta_true, w1):
    bins = np.linspace(0, 2 * np.pi, 61)
    ax.hist(theta_true, bins=bins, color="#888888", alpha=0.6, label="true target")
    ax.hist(theta_gen, bins=bins, color="#1f77b4", alpha=0.6, label="generated")
    ax.set_title(rf"(b) Joint-angle coverage ($W_1$ = {w1:.3f} rad)", fontweight="bold")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("count")
    ax.set_xlim(0, 2 * np.pi)
    ax.legend(fontsize=8)


def plot_branch_coverage(ax, p_gen_plus, js):
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, [BRANCH_PROB_PLUS, 1 - BRANCH_PROB_PLUS], width,
           color="#888888", alpha=0.6, label="true target")
    ax.bar(x + width / 2, [p_gen_plus, 1 - p_gen_plus], width,
           color="#1f77b4", alpha=0.6, label="generated")
    ax.set_xticks(x, ["$b=+1$", "$b=-1$"])
    ax.set_title(f"(c) Branch coverage (JS = {js:.4f})", fontweight="bold")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--checkpoint", default=REPO_ROOT / "checkpoints" / "e00_unconstrained.pkl")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    params = fm.load_params(args.checkpoint)
    rng = jax.random.PRNGKey(args.seed)
    final = np.asarray(fm.sample(rng, params, dim=4, n=args.n_samples, num_steps=100))
    x3, x4 = final[:, :2], final[:, 2:]

    resid = constraint_residuals(x3, x4)
    theta_gen, branch_gen = infer_intrinsic(x3, x4)

    true_theta, true_branch = sample_theta_and_branch(
        np.random.default_rng(args.seed), args.n_samples)
    w1 = wasserstein_1d(theta_gen, true_theta)
    p_gen_plus = float(np.mean(branch_gen == 1))
    js = binary_js_divergence(p_gen_plus, BRANCH_PROB_PLUS)

    print(f"n = {args.n_samples}")
    print(f"constraint error: mean {resid.mean():.4f}, median {np.median(resid):.4f}, "
          f"p99 {np.percentile(resid, 99):.4f}, max {resid.max():.4f}")
    print(f"theta coverage: W1 = {w1:.4f} rad")
    print(f"branch coverage: P(b=+1) generated = {p_gen_plus:.3f}, "
          f"true = {BRANCH_PROB_PLUS:.3f}, JS = {js:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_constraint_error(axes[0], resid)
    plot_theta_coverage(axes[1], theta_gen, true_theta, w1)
    plot_branch_coverage(axes[2], p_gen_plus, js)
    fig.suptitle("Unconstrained baseline, Toy Problem A: error vs. coverage "
                  f"(n={args.n_samples})", fontweight="bold")
    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "E03_e00_metrics.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
