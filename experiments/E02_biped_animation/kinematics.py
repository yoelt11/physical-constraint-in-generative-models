"""Kinematics and gait synthesis for the two-legged biped stick figure.

Topology:
  Fixed hip: x0 = (0, 0)
  Leg A (blue): knee x1, foot x2, link lengths l1A = 1.0, l2A = 0.8
  Leg B (red):  knee x3, foot x4, link lengths l1B = 1.0, l2B = 0.8

Both legs share the fixed hip pivot with no loop-closure constraint (tree topology),
giving 2 independent DOF per leg (4 DOF total).

Kinematic Gait:
  - Antiphase coordination: Leg A phase = phi, Leg B phase = phi + pi
  - Stance phase (phi in [0, pi]): Foot tracks flat ground level (y = -1.6660)
    with backward knee flexion to cushion load and avoid penetration.
  - Swing phase (phi in [pi, 2*pi]): Foot lifts with backward knee flexion
    providing ground clearance of ~0.31 units before touching down.
  - All four link lengths l1A, l2A, l1B, l2B are preserved exactly (residual ~ 10^-16).
"""

import numpy as np

X0 = np.array([0.0, 0.0])  # hip, fixed anchor
L1, L2 = 1.0, 0.8          # thigh, shin

LEG_COLORS = {"A": "#1f77b4", "B": "#d62728"}
GROUND_Y = -1.6660254      # ground level at symmetric double-support pose (x = +/- 0.5)


def leg_kinematics(theta_hip, theta_knee):
    """Knee and foot position for one leg, given its two joint angles (in radians)."""
    knee = X0 + L1 * np.array([np.cos(theta_hip), np.sin(theta_hip)])
    foot = knee + L2 * np.array([np.cos(theta_knee), np.sin(theta_knee)])
    return knee, foot


def get_biped_gait_pose(phi):
    """Return (theta_hip, theta_knee, is_stance) for both legs at gait phase phi (radians).
    
    Leg A has phase phi.
    Leg B has phase (phi + pi), providing antiphase coordination.
    """
    poses = {}
    for leg, p in [("A", phi), ("B", phi + np.pi)]:
        p_mod = p % (2 * np.pi)
        # Hip joint angle oscillation: -90 deg +/- 30 deg
        th_hip = -90.0 + 30.0 * np.cos(p_mod)
        th_h = np.radians(th_hip)
        
        # Stance knee angle to keep foot flush on ground y = GROUND_Y
        knee_y = L1 * np.sin(th_h)
        rho = np.clip((GROUND_Y - knee_y) / L2, -1.0, 1.0)
        th_k_stance = -np.pi - np.arcsin(rho)
        
        if p_mod <= np.pi:
            # Stance phase: foot travels backward along flat ground
            th_knee = th_k_stance
            is_stance = True
        else:
            # Swing phase: knee flexes backward by up to 30 deg to lift foot above ground
            swing_flex = np.radians(30.0) * np.sin(p_mod - np.pi)
            th_knee = th_k_stance - swing_flex
            is_stance = False
            
        poses[leg] = (th_h, th_knee, is_stance)
    return poses


def compute_residuals(k_A, f_A, k_B, f_B):
    """Compute equality constraint violations for all four rigid links."""
    res_1A = abs(np.linalg.norm(k_A - X0) - L1)
    res_2A = abs(np.linalg.norm(f_A - k_A) - L2)
    res_1B = abs(np.linalg.norm(k_B - X0) - L1)
    res_2B = abs(np.linalg.norm(f_B - k_B) - L2)
    return max(res_1A, res_2A, res_1B, res_2B)


def precompute_trajectories(num_points=300):
    """Precompute full-cycle trajectories for visualization overlays."""
    all_phis = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    foot_traj = {"A": [], "B": []}
    knee_traj = {"A": [], "B": []}
    hip_angles = {"A": [], "B": []}
    knee_angles = {"A": [], "B": []}

    for p in all_phis:
        poses = get_biped_gait_pose(p)
        for leg in ["A", "B"]:
            th_h, th_k, _ = poses[leg]
            k, f = leg_kinematics(th_h, th_k)
            foot_traj[leg].append(f)
            knee_traj[leg].append(k)
            hip_angles[leg].append(np.degrees(th_h))
            knee_angles[leg].append(np.degrees(th_k))

    return {
        "phis": all_phis,
        "foot_traj": {k: np.array(v) for k, v in foot_traj.items()},
        "knee_traj": {k: np.array(v) for k, v in knee_traj.items()},
        "hip_angles": {k: np.array(v) for k, v in hip_angles.items()},
        "knee_angles": {k: np.array(v) for k, v in knee_angles.items()},
    }
