"""Static spatial comparison of h3-only guidance at different strengths,
on the n_train=50 checkpoint, with the actual training points overlaid
-- the E06 counterpart to E05_e00_train_vs_generated_n50.png, extended
with the guidance-strength comparison from evaluate_h3_guidance_e00.py.

Usage:
    python3 plot_guidance_spatial_e00.py
"""

import pathlib
import sys

import jax
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import flow_matching as fm  # noqa: E402
from e00_linkage_dataset import solve_linkage_batch  # noqa: E402
from evaluate_h3_guidance_e00 import guidance_grad_fn  # noqa: E402

STRENGTHS = [0.0, 0.5, 8.0]
COLORS = {0.0: "#888888", 0.5: "#2ca02c", 8.0: "#d62728"}
N_PLOT = 500
N_TRAIN = 50


def true_coupler_curves():
    thetas = np.linspace(0, 2 * np.pi, 720)
    x3, x4_plus, x4_minus = solve_linkage_batch(thetas)
    return 0.5 * (x3 + x4_plus), 0.5 * (x3 + x4_minus)


def load_training_coupler_points(data_path, n_train):
    npz = np.load(data_path)
    x3, x4 = npz["x3"][:n_train], npz["x4"][:n_train]
    return 0.5 * (x3 + x4)


def main():
    ckpt_path = REPO_ROOT / "checkpoints" / f"e00_scaling_n{N_TRAIN}.pkl"
    params = fm.load_params(ckpt_path)
    train_coupler = load_training_coupler_points(
        REPO_ROOT / "data" / "e00_linkage_dataset.npz", N_TRAIN)

    samples = {}
    for strength in STRENGTHS:
        rng = jax.random.PRNGKey(2)
        final = np.asarray(fm.guided_sample(rng, params, dim=4, n=N_PLOT,
                                              guidance_grad_fn=guidance_grad_fn,
                                              guidance_strength=strength, num_steps=100))
        samples[strength] = final

    coupler_plus, coupler_minus = true_coupler_curves()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(*coupler_plus.T, color="black", lw=1.2, alpha=0.5, label="true feasible manifold")
    ax.plot(*coupler_minus.T, color="black", lw=1.2, alpha=0.5)
    for strength in STRENGTHS:
        x = samples[strength]
        coupler = 0.5 * (x[:, :2] + x[:, 2:])
        ax.scatter(coupler[:, 0], coupler[:, 1], s=10, alpha=0.5, color=COLORS[strength],
                   label=f"generated, guidance strength={strength}")
    ax.scatter(train_coupler[:, 0], train_coupler[:, 1], s=70, color="#ff7f0e",
               edgecolor="black", linewidth=0.6, zorder=5, label=f"training points (n={N_TRAIN})")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Toy Problem A: h3 guidance vs. training data (n_train=50)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "E06_e00_guidance_spatial.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
