"""Static spatial comparison of h3-only guidance at different strengths,
on the n_train=50 checkpoint -- the E06 counterpart to
E05_e00_train_vs_generated_n50.png.

Two panels, since h3 = ||x4 - x1|| - l41 constrains x4 specifically:
(a) the usual coupler-point view (0.5*(x3+x4)) against the true
    feasible manifold -- the holistic, distributional effect;
(b) x4 alone against the circle ||x4-x1|| = l41 it should lie on if
    h3 = 0 -- the direct, mechanism-specific effect, since panel (a)
    mixes x3 and x4 and could dilute what guidance is actually doing.

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
from e00_linkage_dataset import X1, L41, solve_linkage_batch  # noqa: E402
from evaluate_h3_guidance_e00 import guidance_grad_fn  # noqa: E402

STRENGTHS = [0.0, 0.5, 8.0]
COLORS = {0.0: "#888888", 0.5: "#2ca02c", 8.0: "#d62728"}
N_PLOT = 500


def true_coupler_curves():
    thetas = np.linspace(0, 2 * np.pi, 720)
    x3, x4_plus, x4_minus = solve_linkage_batch(thetas)
    return 0.5 * (x3 + x4_plus), 0.5 * (x3 + x4_minus)


def main():
    ckpt_path = REPO_ROOT / "checkpoints" / "e00_scaling_n50.pkl"
    params = fm.load_params(ckpt_path)

    samples = {}
    for strength in STRENGTHS:
        rng = jax.random.PRNGKey(2)
        final = np.asarray(fm.guided_sample(rng, params, dim=4, n=N_PLOT,
                                              guidance_grad_fn=guidance_grad_fn,
                                              guidance_strength=strength, num_steps=100))
        samples[strength] = final

    coupler_plus, coupler_minus = true_coupler_curves()
    circle_angles = np.linspace(0, 2 * np.pi, 200)
    x4_circle = X1 + L41 * np.c_[np.cos(circle_angles), np.sin(circle_angles)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    ax = axes[0]
    ax.plot(*coupler_plus.T, color="black", lw=1.2, alpha=0.5, label="true feasible manifold")
    ax.plot(*coupler_minus.T, color="black", lw=1.2, alpha=0.5)
    for strength in STRENGTHS:
        x = samples[strength]
        coupler = 0.5 * (x[:, :2] + x[:, 2:])
        ax.scatter(coupler[:, 0], coupler[:, 1], s=10, alpha=0.5, color=COLORS[strength],
                   label=f"guidance strength={strength}")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("(a) Coupler point (0.5*(x3+x4)): holistic effect", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(*x4_circle.T, "--", color="black", lw=1.2, alpha=0.6,
             label=r"$\|x_4-x_1\|=\ell_{41}$ (h3=0)")
    ax.scatter(*X1, color="black", marker="s", s=40, zorder=4)
    for strength in STRENGTHS:
        x4 = samples[strength][:, 2:]
        ax.scatter(x4[:, 0], x4[:, 1], s=10, alpha=0.5, color=COLORS[strength],
                   label=f"guidance strength={strength}")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(r"(b) $x_4$ alone: direct h3-specific effect", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Toy Problem A: h3 guidance, spatial effect (n_train=50)", fontweight="bold")
    fig.tight_layout()
    out_path = REPO_ROOT / "figures" / "E06_e00_guidance_spatial.png"
    fig.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
