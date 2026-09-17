"""Data-scaling study for Toy Problem A: does the unconstrained
flow-matching baseline's constraint satisfaction and distribution
coverage improve with more training data alone -- architecture,
training steps, and batch size all held fixed, only n_train varies --
and where does it start to fail? Real applications rarely have E03's
100,000 clean samples; this checks how much of E04's "projection barely
matters" finding depends on that data abundance.

Usage:
    python3 train_and_evaluate_scaling_e00.py
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
    THETA_MODES, BRANCH_PROB_PLUS, sample_theta_and_branch,
    constraint_residuals, infer_intrinsic,
)

DIM = 4
N_TRAIN_VALUES = [20, 50, 100, 200, 500, 1_000, 10_000, 100_000]
N_EVAL = 5000
STEPS = 20_000


def load_full_dataset(path):
    npz = np.load(path)
    return np.concatenate([npz["x3"], npz["x4"]], axis=-1).astype(np.float32)


def mode_occupancy(theta, tol_std=2.0):
    """Fraction of samples within tol_std standard deviations of each
    target mode's mean -- a direct mode-collapse check that a single
    aggregate W1 number could hide (e.g. spreading mass thinly over two
    modes and missing the third entirely can still give a middling W1)."""
    frac = []
    for mean, std, _ in THETA_MODES:
        d = np.minimum(np.abs(theta - mean), 2 * np.pi - np.abs(theta - mean))
        frac.append(float(np.mean(d < tol_std * std)))
    return frac


def run_one(data_full, n_train, eval_seed):
    ckpt_path = REPO_ROOT / "checkpoints" / f"e00_scaling_n{n_train}.pkl"
    if ckpt_path.exists():
        print(f"  (reusing existing checkpoint {ckpt_path.name})")
        params = fm.load_params(ckpt_path)
        losses = [float("nan")]
    else:
        train_data = data_full[:n_train]
        rng = jax.random.PRNGKey(0)
        init_key, train_key = jax.random.split(rng)
        params = fm.init_params(init_key, dim=DIM)
        params, losses = fm.train(train_key, params, train_data, dim=DIM,
                                   steps=STEPS, batch_size=min(512, n_train), log_every=0)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        fm.save_params(params, ckpt_path)

    sample_rng = jax.random.PRNGKey(eval_seed)
    final = np.asarray(fm.sample(sample_rng, params, dim=DIM, n=N_EVAL, num_steps=100))
    x3, x4 = final[:, :2], final[:, 2:]

    resid = constraint_residuals(x3, x4)
    theta_gen, branch_gen = infer_intrinsic(x3, x4)
    true_theta, _ = sample_theta_and_branch(np.random.default_rng(eval_seed), N_EVAL)
    w1 = wasserstein_1d(theta_gen, true_theta)
    p_gen_plus = float(np.mean(branch_gen == 1))
    js = binary_js_divergence(p_gen_plus, BRANCH_PROB_PLUS)
    occ = mode_occupancy(theta_gen)

    return {
        "n_train": n_train,
        "resid_mean": float(resid.mean()), "resid_median": float(np.median(resid)),
        "resid_p99": float(np.percentile(resid, 99)), "resid_max": float(resid.max()),
        "w1": w1, "branch_js": js, "mode_occupancy": occ,
        "final_loss": float(np.mean(losses[-200:])),
    }


def plot_results(results):
    n_trains = [r["n_train"] for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.plot(n_trains, [r["resid_mean"] for r in results], "o-", label="mean")
    ax.plot(n_trains, [r["resid_median"] for r in results], "s-", label="median")
    ax.plot(n_trains, [r["resid_p99"] for r in results], "^-", label="p99")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training set size")
    ax.set_ylabel(r"$|h(x)|$")
    ax.set_title("(a) Constraint error vs. training set size", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(n_trains, [r["w1"] for r in results], "o-", color="#1f77b4")
    ax.set_xscale("log")
    ax.set_xlabel("training set size")
    ax.set_ylabel(r"$W_1(\theta)$ (rad)")
    ax.set_title("(b) Joint-angle coverage vs. training set size", fontweight="bold")

    ax = axes[1, 0]
    ax.plot(n_trains, [r["branch_js"] for r in results], "o-", color="#d62728")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training set size")
    ax.set_ylabel("branch JS divergence")
    ax.set_title("(c) Branch coverage vs. training set size", fontweight="bold")

    ax = axes[1, 1]
    width = 0.2
    x = np.arange(len(results))
    for i, (mean, std, w) in enumerate(THETA_MODES):
        vals = [r["mode_occupancy"][i] for r in results]
        ax.bar(x + (i - 1) * width, vals, width, label=f"mode {i + 1} (target {w:.2f})")
    ax.set_xticks(x, [str(n) for n in n_trains])
    ax.set_xlabel("training set size")
    ax.set_ylabel("occupancy fraction")
    ax.set_title("(d) Per-mode occupancy (mode-collapse check)", fontweight="bold")
    ax.legend(fontsize=7)

    fig.suptitle("Toy Problem A: unconstrained baseline vs. training set size", fontweight="bold")
    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "E05_e00_data_scaling.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=REPO_ROOT / "data" / "e00_linkage_dataset.npz")
    parser.add_argument("--eval-seed", type=int, default=2)
    args = parser.parse_args()

    data_full = load_full_dataset(args.data)
    results = []
    for n_train in N_TRAIN_VALUES:
        print(f"=== n_train = {n_train} ===")
        r = run_one(data_full, n_train, args.eval_seed)
        results.append(r)
        print(f"  resid: mean {r['resid_mean']:.4f} median {r['resid_median']:.4f} "
              f"p99 {r['resid_p99']:.4f} max {r['resid_max']:.4f}")
        print(f"  W1(theta) = {r['w1']:.4f} rad, branch JS = {r['branch_js']:.4f}")
        print(f"  mode occupancy (target 0.45/0.35/0.20): "
              f"{r['mode_occupancy'][0]:.2f}/{r['mode_occupancy'][1]:.2f}/{r['mode_occupancy'][2]:.2f}")

    plot_results(results)


if __name__ == "__main__":
    main()
