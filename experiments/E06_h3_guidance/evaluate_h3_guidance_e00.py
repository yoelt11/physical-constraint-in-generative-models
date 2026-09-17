"""Inference-time gradient guidance targeting a single constraint, h3
(the x4-x1 rocker link), on the n_train=50 checkpoint from E05 -- the
data-scarce regime where distribution coverage was already found to be
degraded (main.typ's "conditional/semantic" family / the guidance
diagram's "backpropagate a constraint loss" idea, adapted to ambient
coordinates: no decoder here, so the constraint gradient attaches
directly to x_t during sampling).

Sweeps guidance strength (0 = the unguided E05 baseline, for direct
comparison) and checks both evaluation axes: does guidance fix h3
specifically without moving h1/h2, and does it preserve or distort
theta/branch coverage.

Usage:
    python3 evaluate_h3_guidance_e00.py --n-train 50
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
from metrics import wasserstein_1d, binary_js_divergence  # noqa: E402
from e00_linkage_dataset import (  # noqa: E402
    X1, L41, BRANCH_PROB_PLUS, sample_theta_and_branch,
    per_constraint_residuals, infer_intrinsic,
)

STRENGTHS = [0.0, 0.5, 2.0, 8.0]


def h3_squared(x):
    x4 = x[2:]
    return jnp.sum((x4 - X1) ** 2) - L41 ** 2


def h3_energy(x):
    return 0.5 * h3_squared(x) ** 2


guidance_grad_fn = jax.vmap(jax.grad(h3_energy))


def run_one(params, strength, n_eval, eval_seed):
    rng = jax.random.PRNGKey(eval_seed)
    final = np.asarray(fm.guided_sample(rng, params, dim=4, n=n_eval,
                                          guidance_grad_fn=guidance_grad_fn,
                                          guidance_strength=strength, num_steps=100))
    x3, x4 = final[:, :2], final[:, 2:]
    components = per_constraint_residuals(x3, x4)
    theta_hat, branch_hat = infer_intrinsic(x3, x4)
    true_theta, _ = sample_theta_and_branch(np.random.default_rng(eval_seed), n_eval)
    w1 = wasserstein_1d(theta_hat, true_theta)
    p_plus = float(np.mean(branch_hat == 1))
    js = binary_js_divergence(p_plus, BRANCH_PROB_PLUS)
    return {"strength": strength, "components": {k: np.abs(v) for k, v in components.items()},
            "w1": w1, "branch_js": js}


def plot_results(results):
    strengths = [r["strength"] for r in results]
    labels = list(results[0]["components"].keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    for k in labels:
        ax.plot(strengths, [r["components"][k].mean() for r in results], "o-", label=f"{k} (mean)")
    ax.set_yscale("log")
    ax.set_xlabel("h3 guidance strength")
    ax.set_ylabel(r"$|h_i(x)|$")
    ax.set_title("(a) Per-constraint error vs. guidance strength", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    data = [r["components"]["h3 (x4-x1 length)"] for r in results]
    ax.boxplot(data, tick_labels=[str(s) for s in strengths], showfliers=True,
               flierprops=dict(marker=".", markersize=3, alpha=0.3))
    ax.set_yscale("log")
    ax.set_xlabel("h3 guidance strength")
    ax.set_ylabel(r"$|h_3(x)|$")
    ax.set_title("(b) h3 distribution vs. guidance strength", fontweight="bold")

    ax = axes[1, 0]
    ax.plot(strengths, [r["w1"] for r in results], "o-", color="#1f77b4")
    ax.set_xlabel("h3 guidance strength")
    ax.set_ylabel(r"$W_1(\theta)$ (rad)")
    ax.set_title("(c) Joint-angle coverage vs. guidance strength", fontweight="bold")

    ax = axes[1, 1]
    ax.plot(strengths, [r["branch_js"] for r in results], "o-", color="#d62728")
    ax.set_yscale("log")
    ax.set_xlabel("h3 guidance strength")
    ax.set_ylabel("branch JS divergence")
    ax.set_title("(d) Branch coverage vs. guidance strength", fontweight="bold")

    fig.suptitle("h3-only inference-time guidance on the n_train=50 baseline", fontweight="bold")
    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "E06_e00_h3_guidance.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=5000)
    parser.add_argument("--eval-seed", type=int, default=2)
    args = parser.parse_args()

    ckpt_path = REPO_ROOT / "checkpoints" / f"e00_scaling_n{args.n_train}.pkl"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found -- run E05's scaling sweep first.")
    params = fm.load_params(ckpt_path)

    results = []
    for strength in STRENGTHS:
        r = run_one(params, strength, args.n_eval, args.eval_seed)
        results.append(r)
        comp_str = ", ".join(f"{k.split(' ')[0]}={v.mean():.4f}" for k, v in r["components"].items())
        print(f"strength={strength}: {comp_str}, W1={r['w1']:.4f}, branch_js={r['branch_js']:.4f}")

    plot_results(results)


if __name__ == "__main__":
    main()
