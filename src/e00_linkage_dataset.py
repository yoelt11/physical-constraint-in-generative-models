"""Dataset generator for Toy Problem A: the four-bar linkage (main.typ,
Section 2). Every sample is built directly from the forward-kinematics
formula, so h(x) = 0 holds by construction -- there is nothing to
reject or filter. Mirrors the parameters and target distribution in
experiments/E00_problem_visualization/visualize_problem.py.

Usage:
    python3 e00_linkage_dataset.py --n-samples 100000 --seed 0
"""

import argparse
import pathlib

import numpy as np

X1 = np.array([0.0, 0.0])
L12, L23, L34, L41 = 1.0, 0.6, 1.3, 0.9
X2 = np.array([L12, 0.0])

# The designed, deliberately multimodal target distribution over the
# intrinsic joint angle theta, and the (imbalanced) branch probability --
# see main.typ's "controllable, multimodal target distribution over
# poses" and the discussion of why satisfying h(x) = 0 alone is not
# sufficient to recover it.
THETA_MODES = [(0.9, 0.20, 0.45), (3.3, 0.15, 0.35), (5.3, 0.25, 0.20)]
BRANCH_PROB_PLUS = 0.7


def sample_theta_and_branch(rng, n):
    means, stds, weights = zip(*THETA_MODES)
    idx = rng.choice(len(THETA_MODES), size=n, p=weights)
    theta = rng.normal(np.array(means)[idx], np.array(stds)[idx])
    branch = rng.choice([1, -1], size=n, p=[BRANCH_PROB_PLUS, 1 - BRANCH_PROB_PLUS])
    return np.mod(theta, 2 * np.pi), branch


def solve_linkage_batch(theta):
    """Vectorized forward kinematics: theta (n,) -> x3 (n,2), x4_plus (n,2),
    x4_minus (n,2). Both branches exist for every theta at these link
    lengths (verified in experiments/E00_problem_visualization)."""
    x3 = X2 + L23 * np.stack([np.cos(theta), np.sin(theta)], axis=-1)

    c0, r0, c1, r1 = X1, L41, x3, L34
    diff = c1 - c0
    d = np.linalg.norm(diff, axis=-1)
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(np.clip(r0 ** 2 - a ** 2, 0.0, None))
    mid = c0 + (a / d)[:, None] * diff
    perp = np.stack([-diff[:, 1], diff[:, 0]], axis=-1) / d[:, None]
    return x3, mid + h[:, None] * perp, mid - h[:, None] * perp


def per_constraint_residuals(x3, x4):
    """h1, h2, h3 evaluated separately (not aggregated), for any (x3, x4) --
    including model output that doesn't exactly satisfy them."""
    h1 = np.linalg.norm(x3 - X2, axis=-1) - L23
    h2 = np.linalg.norm(x4 - x3, axis=-1) - L34
    h3 = np.linalg.norm(x4 - X1, axis=-1) - L41
    return {"h1 (x3-x2 length)": h1, "h2 (x4-x3 length)": h2, "h3 (x4-x1 length)": h3}


def constraint_residuals(x3, x4):
    """Aggregate max|h_i(x)| over the three constraints."""
    components = per_constraint_residuals(x3, x4)
    return np.max(np.abs(np.stack(list(components.values()))), axis=0)


def infer_intrinsic(x3, x4):
    """Recover (theta_hat, branch_hat) from any (x3, x4) -- including
    model output that doesn't exactly satisfy h(x) = 0 -- by inverting
    the forward kinematics: theta_hat from the angle of x3 relative to
    x2, branch_hat from which side of the circle-intersection line x4
    falls on (the same mid/perp construction as solve_linkage_batch)."""
    theta_hat = np.mod(np.arctan2(x3[:, 1] - X2[1], x3[:, 0] - X2[0]), 2 * np.pi)
    diff = x3 - X1
    d = np.linalg.norm(diff, axis=-1)
    a = (L41 ** 2 - L34 ** 2 + d ** 2) / (2 * d)
    mid = X1 + (a / d)[:, None] * diff
    perp = np.stack([-diff[:, 1], diff[:, 0]], axis=-1) / d[:, None]
    side = np.sum((x4 - mid) * perp, axis=-1)
    branch_hat = np.where(side >= 0, 1, -1)
    return theta_hat, branch_hat


def generate_dataset(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    theta, branch = sample_theta_and_branch(rng, n_samples)
    x3, x4_plus, x4_minus = solve_linkage_batch(theta)
    x4 = np.where((branch == 1)[:, None], x4_plus, x4_minus)

    dataset = {
        "x1": np.broadcast_to(X1, (n_samples, 2)).copy(),
        "x2": np.broadcast_to(X2, (n_samples, 2)).copy(),
        "x3": x3,
        "x4": x4,
        "theta": theta,
        "branch": branch,
        "link_lengths": np.array([L12, L23, L34, L41]),
    }
    check_constraints(dataset)
    return dataset


def check_constraints(dataset):
    max_residual = constraint_residuals(dataset["x3"], dataset["x4"]).max()
    assert max_residual < 1e-8, f"constraint residual too large: {max_residual}"
    print(f"  max |h(x)| residual: {max_residual:.2e} (should be ~0, by construction)")


def save_dataset(dataset, out_path):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    print(f"Saved {dataset['theta'].shape[0]} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=pathlib.Path(__file__).resolve().parent.parent / "data" / "e00_linkage_dataset.npz",
    )
    args = parser.parse_args()

    dataset = generate_dataset(args.n_samples, args.seed)
    save_dataset(dataset, args.output)
