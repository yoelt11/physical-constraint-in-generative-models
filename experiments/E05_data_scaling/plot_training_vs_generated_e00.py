"""Static comparison: which training points a small dataset actually
contains, and what the model trained on them generates, against the
true feasible manifold. A static counterpart to the final frame of
figures/E03_e00_unconstrained_flow.gif, but for a specific (small)
n_train instead of the full 100,000-sample baseline -- to show directly
why a sparse training set produces the theta-coverage gaps measured in
train_and_evaluate_scaling_e00.py (e.g. if the training points happen
to under-sample a mode, the generated distribution mirrors that gap).

Usage:
    python3 plot_training_vs_generated_e00.py --n-train 50
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
from e00_linkage_dataset import solve_linkage_batch  # noqa: E402


def load_full_dataset(path):
    npz = np.load(path)
    return np.concatenate([npz["x3"], npz["x4"]], axis=-1).astype(np.float32)


def true_coupler_curves():
    thetas = np.linspace(0, 2 * np.pi, 720)
    x3, x4_plus, x4_minus = solve_linkage_batch(thetas)
    return 0.5 * (x3 + x4_plus), 0.5 * (x3 + x4_minus)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-generated", type=int, default=600)
    parser.add_argument("--data", default=REPO_ROOT / "data" / "e00_linkage_dataset.npz")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    ckpt_path = pathlib.Path(args.checkpoint) if args.checkpoint else (
        REPO_ROOT / "checkpoints" / f"e00_scaling_n{args.n_train}.pkl")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found -- run train_and_evaluate_scaling_e00.py with "
            f"n_train={args.n_train} in N_TRAIN_VALUES first.")
    params = fm.load_params(ckpt_path)

    data_full = load_full_dataset(args.data)
    train_data = data_full[:args.n_train]
    train_coupler = 0.5 * (train_data[:, :2] + train_data[:, 2:])

    rng = jax.random.PRNGKey(args.seed)
    generated = np.asarray(fm.sample(rng, params, dim=4, n=args.n_generated, num_steps=100))
    gen_coupler = 0.5 * (generated[:, :2] + generated[:, 2:])

    coupler_plus, coupler_minus = true_coupler_curves()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(*coupler_plus.T, color="black", lw=1.2, alpha=0.5, label="true feasible manifold")
    ax.plot(*coupler_minus.T, color="black", lw=1.2, alpha=0.5)
    ax.scatter(gen_coupler[:, 0], gen_coupler[:, 1], s=10, color="#1f77b4", alpha=0.5,
               label=f"generated (n={args.n_generated})")
    ax.scatter(train_coupler[:, 0], train_coupler[:, 1], s=70, color="#ff7f0e",
               edgecolor="black", linewidth=0.6, zorder=5,
               label=f"training points (n={args.n_train})")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(f"Toy Problem A: training data vs. generated samples "
                  f"(n_train = {args.n_train})", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / f"E05_e00_train_vs_generated_n{args.n_train}.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
