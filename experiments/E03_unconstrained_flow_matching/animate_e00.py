"""Animate the unconstrained flow-matching baseline for Toy Problem A:
the ODE integration from noise to generated samples, overlaid on the
true coupler curve. This is the motivating visual for constraint
enforcement -- many samples drift toward, but do not land on, the valid
manifold, since nothing in training or sampling enforces h(x) = 0.

Usage:
    python3 animate_e00.py
"""

import pathlib
import sys

import jax
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import flow_matching as fm  # noqa: E402
from e00_linkage_dataset import X1, X2, L23, L34, L41, solve_linkage_batch  # noqa: E402

N_SAMPLES = 600
NUM_STEPS = 60
FPS = 25
# Residual, over many samples, is not cleanly bimodal (most points land
# close to the manifold, but with a heavy tail of clear failures) -- so
# points are colored by a continuous log10(residual) scale rather than
# a binary valid/invalid split, which would either hide the tail or look
# like an arbitrarily chosen cutoff.
LOG_RESID_RANGE = (-3.0, -0.3)  # residual from 0.001 to ~0.5


def residuals(x3, x4):
    h1 = np.linalg.norm(x3 - X2, axis=-1) - L23
    h2 = np.linalg.norm(x4 - x3, axis=-1) - L34
    h3 = np.linalg.norm(x4 - X1, axis=-1) - L41
    return np.max(np.abs(np.stack([h1, h2, h3])), axis=0)


def true_coupler_curves():
    thetas = np.linspace(0, 2 * np.pi, 720)
    x3, x4_plus, x4_minus = solve_linkage_batch(thetas)
    return 0.5 * (x3 + x4_plus), 0.5 * (x3 + x4_minus)


def main():
    ckpt_path = REPO_ROOT / "checkpoints" / "e00_unconstrained.pkl"
    params = fm.load_params(ckpt_path)

    rng = jax.random.PRNGKey(1)
    _, trajectory = fm.sample(rng, params, dim=4, n=N_SAMPLES, num_steps=NUM_STEPS,
                               return_trajectory=True)
    trajectory = np.asarray(trajectory)  # (NUM_STEPS + 1, N_SAMPLES, 4)

    final_x3, final_x4 = trajectory[-1, :, :2], trajectory[-1, :, 2:]
    resid = residuals(final_x3, final_x4)
    log_resid = np.log10(np.clip(resid, 10 ** LOG_RESID_RANGE[0], None))
    pct = np.percentile(resid, [50, 90, 99, 100])
    print(f"residual over {N_SAMPLES} unconstrained samples -- "
          f"median {pct[0]:.4f}, p90 {pct[1]:.4f}, p99 {pct[2]:.4f}, max {pct[3]:.4f}")

    coupler_plus, coupler_minus = true_coupler_curves()
    coupler_traj = 0.5 * (trajectory[:, :, :2] + trajectory[:, :, 2:])  # (T, N, 2)

    lim = 2.2
    frames = []
    for frame_i in range(trajectory.shape[0]):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(*coupler_plus.T, color="black", lw=1.2, alpha=0.5, label="true feasible manifold")
        ax.plot(*coupler_minus.T, color="black", lw=1.2, alpha=0.5)
        pts = coupler_traj[frame_i]
        sc = ax.scatter(pts[:, 0], pts[:, 1], s=12, c=log_resid, cmap="RdYlGn_r",
                        vmin=LOG_RESID_RANGE[0], vmax=LOG_RESID_RANGE[1], alpha=0.8)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"$log_{10} |h(x)|$ (at $t=1$)")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        t = frame_i / NUM_STEPS
        ax.set_title(f"Unconstrained flow matching, Toy Problem A ($t={t:.2f}$)", fontweight="bold")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())))
        plt.close(fig)

    out_path = REPO_ROOT / "figures" / "E03_e00_unconstrained_flow.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hold = [frames[-1]] * FPS  # pause ~1s on the final, most informative frame
    frames[0].save(out_path, save_all=True, append_images=frames[1:] + hold,
                    duration=int(1000 / FPS), loop=0)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
