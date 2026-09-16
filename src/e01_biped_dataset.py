"""Dataset generator for Toy Problem B: the two-legged biped (main.typ,
Section 3). Every sample is built directly from the forward-kinematics
formula (no loop closure, so no circle-intersection / branch choice is
needed), so h(x) = 0 holds by construction. Mirrors the parameters in
experiments/E01_problem_visualization/kinematics.py.

The four joint angles are currently sampled *independently and
uniformly* over a plausible "legs hang below the hip" range -- this is
a placeholder, not a designed target distribution. main.typ Section 3
and Section 5 ("Conditional or semantic constraints") flag gait-phase
coupling, ground contact, foot-foot collision, and knee-bend limits as
open extensions; none of that is implemented here yet, so this dataset
should not be read as "realistic walking poses," only as valid
(constraint-satisfying) ones. --dump-diagnostics reports, purely for
information, how often a naively-sampled pose would already violate
those not-yet-enforced constraints.

Usage:
    python3 e01_biped_dataset.py --n-samples 100000 --seed 0
"""

import argparse
import pathlib

import numpy as np

X0 = np.array([0.0, 0.0])
L1, L2 = 1.0, 0.8  # thigh, shin -- shared by both legs

# Placeholder sampling range per joint angle (degrees): legs pointing
# generally downward from the hip, not wrapping back up through it.
ANGLE_RANGE_DEG = (200.0, 340.0)


def sample_angles(rng, n_samples):
    lo, hi = np.radians(ANGLE_RANGE_DEG)
    shape = (n_samples,)
    return {
        "theta_hip_A": rng.uniform(lo, hi, shape),
        "theta_knee_A": rng.uniform(lo, hi, shape),
        "theta_hip_B": rng.uniform(lo, hi, shape),
        "theta_knee_B": rng.uniform(lo, hi, shape),
    }


def leg_kinematics_batch(theta_hip, theta_knee):
    knee = X0 + L1 * np.stack([np.cos(theta_hip), np.sin(theta_hip)], axis=-1)
    foot = knee + L2 * np.stack([np.cos(theta_knee), np.sin(theta_knee)], axis=-1)
    return knee, foot


def generate_dataset(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    angles = sample_angles(rng, n_samples)
    knee_A, foot_A = leg_kinematics_batch(angles["theta_hip_A"], angles["theta_knee_A"])
    knee_B, foot_B = leg_kinematics_batch(angles["theta_hip_B"], angles["theta_knee_B"])

    dataset = {
        "x0": np.broadcast_to(X0, (n_samples, 2)).copy(),
        "knee_A": knee_A,
        "foot_A": foot_A,
        "knee_B": knee_B,
        "foot_B": foot_B,
        "link_lengths": np.array([L1, L2]),
        **angles,
    }
    check_constraints(dataset)
    return dataset


def check_constraints(dataset):
    x0 = dataset["x0"]
    h1a = np.linalg.norm(dataset["knee_A"] - x0, axis=-1) - L1
    h2a = np.linalg.norm(dataset["foot_A"] - dataset["knee_A"], axis=-1) - L2
    h1b = np.linalg.norm(dataset["knee_B"] - x0, axis=-1) - L1
    h2b = np.linalg.norm(dataset["foot_B"] - dataset["knee_B"], axis=-1) - L2
    max_residual = np.max(np.abs(np.stack([h1a, h2a, h1b, h2b])))
    assert max_residual < 1e-8, f"constraint residual too large: {max_residual}"
    print(f"  max |h(x)| residual: {max_residual:.2e} (should be ~0, by construction)")


def report_unenforced_constraint_diagnostics(dataset, d_min=0.3):
    """Informational only -- not enforced or filtered by this generator.

    No ground-plane level is defined for this toy problem yet (the hip
    x0 sits at the origin and both legs hang into negative y by
    sampling-range construction, so a naive "y < 0" check would be
    ~100% trivially and would say nothing about an actual ground
    constraint); only the foot-foot collision check below is currently
    well-defined without further design decisions.
    """
    foot_A, foot_B = dataset["foot_A"], dataset["foot_B"]
    foot_dist = np.linalg.norm(foot_A - foot_B, axis=-1)
    colliding = foot_dist < d_min

    print("  (informational; not enforced by this generator)")
    print(f"  foot-foot distance < {d_min}: {colliding.mean():.1%}")


def save_dataset(dataset, out_path):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    n = dataset["x0"].shape[0]
    print(f"Saved {n} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=pathlib.Path(__file__).resolve().parent.parent / "data" / "e01_biped_dataset.npz",
    )
    parser.add_argument("--dump-diagnostics", action="store_true",
                         help="Print informational stats on ground/collision violations.")
    args = parser.parse_args()

    dataset = generate_dataset(args.n_samples, args.seed)
    if args.dump_diagnostics:
        report_unenforced_constraint_diagnostics(dataset)
    save_dataset(dataset, args.output)
