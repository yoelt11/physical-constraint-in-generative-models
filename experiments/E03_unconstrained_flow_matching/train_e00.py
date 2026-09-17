"""Train an unconstrained flow-matching baseline on Toy Problem A (the
four-bar linkage). No constraint enforcement of any kind -- this is the
motivating baseline: see animate_e00.py for where the generated samples
land relative to the true feasible manifold.

Usage:
    python3 train_e00.py --steps 20000
"""

import argparse
import pathlib
import sys

import jax
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import flow_matching as fm  # noqa: E402

DIM = 4  # (x3, x4) in R^2 x R^2


def load_dataset(path):
    npz = np.load(path)
    return np.concatenate([npz["x3"], npz["x4"]], axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=REPO_ROOT / "data" / "e00_linkage_dataset.npz")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=REPO_ROOT / "checkpoints" / "e00_unconstrained.pkl")
    args = parser.parse_args()

    data = load_dataset(args.data)
    print(f"Loaded {data.shape[0]} samples, dim={data.shape[1]}")

    rng = jax.random.PRNGKey(args.seed)
    init_key, train_key = jax.random.split(rng)
    params = fm.init_params(init_key, dim=DIM)
    params, losses = fm.train(
        train_key, params, data, dim=DIM,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
    )

    ckpt_path = pathlib.Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    fm.save_params(params, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(losses)
    ax.set_xlabel("step")
    ax.set_ylabel("flow-matching loss")
    ax.set_title("Toy Problem A: unconstrained baseline training loss")
    fig.tight_layout()
    fig_dir = REPO_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "E03_e00_training_loss.png", dpi=150)
    print(f"Saved {fig_dir / 'E03_e00_training_loss.png'}")


if __name__ == "__main__":
    main()
