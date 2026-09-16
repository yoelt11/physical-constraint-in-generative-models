"""Animated visualization of the two-legged biped stick figure.

Generates high-quality looping GIF animations:
1. E02_biped_walking_gait.gif:
   Multi-panel layout showing:
   - Panel (a): Physical mechanism with fixed hip x0, legs A & B, reachable annulus,
     flat ground contact, trailing foot trajectories, and live constraint residuals.
   - Panel (b): Configuration space torus T^2 (theta_hip, theta_knee) with live limit-cycle state.
   - Panel (c): Stride profile & ground clearance y_foot(phi) across the gait cycle.

2. E02_biped_walking_focus.gif:
   Clean single-panel animation matching E01_mechanism_schematic.png with full labels
   (x0, x1, x2, x3, x4, l1A, l2A, l1B, l2B), ground line, and smooth walking gait.
"""

import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from PIL import Image

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from kinematics import (
    X0, L1, L2, LEG_COLORS, GROUND_Y,
    leg_kinematics, get_biped_gait_pose,
    compute_residuals, precompute_trajectories
)

# Apply style if available
style_path = pathlib.Path("/home/etorres/Documents/github/personal/research-project-pinns/style/pitayasmoothie-light.mplstyle")
if style_path.exists():
    plt.style.use(str(style_path))

ANCHOR_COLOR = "#2b2b2b"
GROUND_COLOR = "#555555"
NUM_FRAMES = 90
FPS = 25


def draw_ground_hatching(ax, xmin=-1.25, xmax=1.25, y=GROUND_Y, dy=0.06, spacing=0.08):
    """Draw ground level line and mechanical ground hatching."""
    ax.axhline(y, color=GROUND_COLOR, lw=2.2, zorder=2)
    xs = np.arange(xmin, xmax, spacing)
    for x in xs:
        ax.plot([x, x - dy * 0.7], [y, y - dy], color=GROUND_COLOR, lw=1.2, zorder=2)


def render_gait_frame(phi, idx, traj_data):
    """Render a single frame for the multi-panel gait animation."""
    poses = get_biped_gait_pose(phi)
    deg_phi = np.degrees(phi) % 360
    
    fig = plt.figure(figsize=(15, 7.2), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1, 1],
                          left=0.07, right=0.96, top=0.92, bottom=0.09,
                          wspace=0.28, hspace=0.35)
    
    ax_mech = fig.add_subplot(gs[:, 0])
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, 1])
    
    for ax in [ax_mech, ax_phase, ax_trace]:
        ax.set_facecolor("#fafafa")
        ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    
    # ---------------- 1. Main Mechanism Panel ----------------
    ax_mech.set_aspect("equal")
    ax_mech.set_xlim(-1.25, 1.25)
    ax_mech.set_ylim(-2.05, 0.25)
    
    # Annular workspace and knee orbit
    outer_r = L1 + L2
    inner_r = abs(L1 - L2)
    ax_mech.add_patch(Wedge(X0, outer_r, 0, 360, width=outer_r - inner_r,
                            facecolor="#e2e8f0", alpha=0.45, edgecolor="none", zorder=1))
    ax_mech.add_patch(Circle(X0, L1, fill=False, linestyle=":", color="#a0aec0", lw=1.2, zorder=1))
    
    # Ground line & label
    draw_ground_hatching(ax_mech, -1.25, 1.25, GROUND_Y)
    ax_mech.text(1.18, GROUND_Y + 0.04, "ground level", fontsize=9, color=GROUND_COLOR,
                 ha="right", va="bottom", fontweight="bold")
    
    # Static full foot trajectories (faint)
    for leg in ["A", "B"]:
        ax_mech.plot(traj_data["foot_traj"][leg][:, 0], traj_data["foot_traj"][leg][:, 1],
                     "--", color=LEG_COLORS[leg], alpha=0.3, lw=1.5, zorder=2)
    
    # Active trailing path up to current frame
    curr_sub_idx = int((phi / (2 * np.pi)) * len(traj_data["phis"])) % len(traj_data["phis"])
    for leg in ["A", "B"]:
        ax_mech.plot(traj_data["foot_traj"][leg][:curr_sub_idx+1, 0],
                     traj_data["foot_traj"][leg][:curr_sub_idx+1, 1],
                     "-", color=LEG_COLORS[leg], alpha=0.85, lw=2.5, zorder=3)
    
    # Draw legs
    leg_coords = {}
    for leg in ["A", "B"]:
        th_h, th_k, is_stance = poses[leg]
        knee, foot = leg_kinematics(th_h, th_k)
        leg_coords[leg] = (knee, foot, is_stance)
        color = LEG_COLORS[leg]
        knee_idx, foot_idx = ("1", "2") if leg == "A" else ("3", "4")
        
        # Thigh link
        ax_mech.plot([X0[0], knee[0]], [X0[1], knee[1]], "-", color=color, lw=4.2,
                     solid_capstyle="round", zorder=4)
        # Shin link
        ax_mech.plot([knee[0], foot[0]], [knee[1], foot[1]], "-", color=color, lw=4.0,
                     solid_capstyle="round", zorder=4)
        
        # Nodes
        ax_mech.scatter(*knee, color=color, marker="o", s=130, edgecolors="white", lw=1.5, zorder=6)
        ax_mech.scatter(*foot, color=color, marker="o", s=130, edgecolors="white", lw=1.5, zorder=6)
        
        # Text labels
        knee_x_off = 10 if leg == "A" else -12
        knee_ha = "left" if leg == "A" else "right"
        ax_mech.annotate(rf"$x_{{{knee_idx}}}$ (knee {leg})", knee,
                         textcoords="offset points", xytext=(knee_x_off, 6),
                         fontsize=11, fontweight="bold", color=color, ha=knee_ha)
        
        foot_x_off = 10 if leg == "A" else -12
        foot_ha = "left" if leg == "A" else "right"
        ax_mech.annotate(rf"$x_{{{foot_idx}}}$ (foot {leg})", foot,
                         textcoords="offset points", xytext=(foot_x_off, -16),
                         fontsize=11, fontweight="bold", color=color, ha=foot_ha)
    
    # Hip anchor
    ax_mech.scatter(*X0, color=ANCHOR_COLOR, marker="s", s=160, zorder=7)
    ax_mech.annotate(r"$x_0$ (hip, fixed)", X0, textcoords="offset points",
                     xytext=(10, 8), fontsize=12, fontweight="bold", color=ANCHOR_COLOR)
    
    # Compute residuals
    k_A, f_A, st_A = leg_coords["A"]
    k_B, f_B, st_B = leg_coords["B"]
    max_res = compute_residuals(k_A, f_A, k_B, f_B)
    
    badge_text = (
        rf"$\phi = {deg_phi:5.1f}^\circ$" + "\n"
        rf"Leg A: {'STANCE' if st_A else 'SWING '}" + "\n"
        rf"Leg B: {'STANCE' if st_B else 'SWING '}" + "\n"
        rf"$\max \|h\| = {max_res:.1e}$"
    )
    ax_mech.text(0.04, 0.96, badge_text, transform=ax_mech.transAxes, fontsize=10,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.92))
    
    ax_mech.set_title("Two-Legged Biped Kinematics: Antiphase Walking Gait", fontsize=13, fontweight="bold", pad=10)
    ax_mech.set_xlabel("$x$", fontsize=11)
    ax_mech.set_ylabel("$y$", fontsize=11)
    
    # ---------------- 2. Configuration Space Panel ----------------
    ax_phase.plot(traj_data["hip_angles"]["A"], traj_data["knee_angles"]["A"],
                  "-", color=LEG_COLORS["A"], lw=2.2, label="Leg A limit cycle")
    ax_phase.plot(traj_data["hip_angles"]["B"], traj_data["knee_angles"]["B"],
                  "--", color=LEG_COLORS["B"], lw=2.2, label="Leg B limit cycle")
    
    curr_h_A, curr_k_A = np.degrees(poses["A"][0]), np.degrees(poses["A"][1])
    curr_h_B, curr_k_B = np.degrees(poses["B"][0]), np.degrees(poses["B"][1])
    ax_phase.scatter(curr_h_A, curr_k_A, color=LEG_COLORS["A"], s=120, edgecolors="white", lw=1.5, zorder=5)
    ax_phase.scatter(curr_h_B, curr_k_B, color=LEG_COLORS["B"], s=120, edgecolors="white", lw=1.5, zorder=5)
    
    ax_phase.set_title(r"Configuration Space: Torus $T^2$ $(\theta_{\rm hip}, \theta_{\rm knee})$", fontsize=11, fontweight="bold")
    ax_phase.set_xlabel(r"Hip Angle $\theta_{\rm hip}$ (deg)", fontsize=10)
    ax_phase.set_ylabel(r"Knee Angle $\theta_{\rm knee}$ (deg)", fontsize=10)
    ax_phase.legend(loc="lower left", fontsize=8.5)
    
    # ---------------- 3. Stride & Clearance Panel ----------------
    phis_deg = np.degrees(traj_data["phis"])
    feet_y_A = traj_data["foot_traj"]["A"][:, 1]
    feet_y_B = traj_data["foot_traj"]["B"][:, 1]
    
    ax_trace.plot(phis_deg, feet_y_A, "-", color=LEG_COLORS["A"], lw=2.0, label=r"Foot A height $y_2$")
    ax_trace.plot(phis_deg, feet_y_B, "-", color=LEG_COLORS["B"], lw=2.0, label=r"Foot B height $y_4$")
    ax_trace.axhline(GROUND_Y, color=GROUND_COLOR, linestyle=":", lw=1.5, label="Ground level")
    
    ax_trace.axvline(deg_phi, color="#ff7f0e", linestyle="-", lw=1.8, alpha=0.8)
    ax_trace.scatter(deg_phi, f_A[1], color=LEG_COLORS["A"], s=70, zorder=5)
    ax_trace.scatter(deg_phi, f_B[1], color=LEG_COLORS["B"], s=70, zorder=5)
    
    ax_trace.set_title("Foot Ground Clearance & Periodic Stride Profile", fontsize=11, fontweight="bold")
    ax_trace.set_xlabel(r"Gait Cycle Phase $\phi$ (deg)", fontsize=10)
    ax_trace.set_ylabel(r"Vertical Position $y$", fontsize=10)
    ax_trace.set_xlim(0, 360)
    ax_trace.set_ylim(-1.85, -1.25)
    ax_trace.legend(loc="lower right", fontsize=8)
    
    fig.suptitle(rf"Two-Legged Biped Stick Figure: Tree Topology & Constrained Motion ($\phi = {deg_phi:5.1f}^\circ$)",
                 fontsize=14, fontweight="bold", y=0.98)
    
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba)
    plt.close(fig)
    return img


def render_focus_frame(phi, idx, traj_data):
    """Render a clean single-panel animation matching E01_mechanism_schematic.png."""
    poses = get_biped_gait_pose(phi)
    deg_phi = np.degrees(phi) % 360
    
    fig, ax = plt.subplots(figsize=(7.5, 7.8), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.95, 0.25)
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    
    # Ground line & hatching
    draw_ground_hatching(ax, -1.15, 1.15, GROUND_Y)
    
    # Trailing foot path
    curr_sub_idx = int((phi / (2 * np.pi)) * len(traj_data["phis"])) % len(traj_data["phis"])
    for leg in ["A", "B"]:
        ax.plot(traj_data["foot_traj"][leg][:, 0], traj_data["foot_traj"][leg][:, 1],
                "--", color=LEG_COLORS[leg], alpha=0.3, lw=1.5, zorder=2)
        ax.plot(traj_data["foot_traj"][leg][:curr_sub_idx+1, 0],
                traj_data["foot_traj"][leg][:curr_sub_idx+1, 1],
                "-", color=LEG_COLORS[leg], alpha=0.85, lw=2.5, zorder=3)
    
    for leg in ["A", "B"]:
        th_h, th_k, is_stance = poses[leg]
        knee, foot = leg_kinematics(th_h, th_k)
        color = LEG_COLORS[leg]
        knee_idx, foot_idx = ("1", "2") if leg == "A" else ("3", "4")
        
        # Links
        ax.plot([X0[0], knee[0]], [X0[1], knee[1]], "-", color=color, lw=4.2,
                solid_capstyle="round", zorder=4)
        ax.plot([knee[0], foot[0]], [knee[1], foot[1]], "-", color=color, lw=4.0,
                solid_capstyle="round", zorder=4)
        
        # Link annotations at midpoint
        mid_thigh = 0.5 * (X0 + knee)
        mid_shin = 0.5 * (knee + foot)
        ax.annotate(rf"$\ell_{{1{leg}}}$", mid_thigh, textcoords="offset points",
                    xytext=(6, 4), fontsize=11, color=color, fontweight="bold")
        ax.annotate(rf"$\ell_{{2{leg}}}$", mid_shin, textcoords="offset points",
                    xytext=(6, 0), fontsize=11, color=color, fontweight="bold")
        
        # Nodes
        ax.scatter(*knee, color=color, marker="o", s=140, edgecolors="white", lw=1.5, zorder=6)
        ax.scatter(*foot, color=color, marker="o", s=140, edgecolors="white", lw=1.5, zorder=6)
        
        knee_x_off = 10 if leg == "A" else -12
        knee_ha = "left" if leg == "A" else "right"
        ax.annotate(rf"$x_{{{knee_idx}}}$ (knee {leg})", knee,
                    textcoords="offset points", xytext=(knee_x_off, 8),
                    fontsize=11, fontweight="bold", color=color, ha=knee_ha)
        
        foot_x_off = 10 if leg == "A" else -12
        foot_ha = "left" if leg == "A" else "right"
        ax.annotate(rf"$x_{{{foot_idx}}}$ (foot {leg})", foot,
                    textcoords="offset points", xytext=(foot_x_off, -16),
                    fontsize=11, fontweight="bold", color=color, ha=foot_ha)
    
    # Hip anchor
    ax.scatter(*X0, color=ANCHOR_COLOR, marker="s", s=160, zorder=7, label="anchor (fixed)")
    ax.annotate(r"$x_0$ (hip)", X0, textcoords="offset points", xytext=(10, 8),
                fontsize=12, fontweight="bold", color=ANCHOR_COLOR)
    
    # Invisible proxies for legend
    ax.scatter([], [], color=LEG_COLORS["A"], marker="o", s=120, label="leg A (free)")
    ax.scatter([], [], color=LEG_COLORS["B"], marker="o", s=120, label="leg B (free)")
    
    ax.set_title(rf"Two-Legged Biped Stick Figure: Walking Gait ($\phi = {deg_phi:5.1f}^\circ$)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$y$", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=9.5)
    
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba)
    plt.close(fig)
    return img


def main():
    print(f"Precomputing biped trajectories ({NUM_FRAMES} frames)...")
    traj_data = precompute_trajectories(300)
    thetas_all = np.linspace(0, 2 * np.pi, NUM_FRAMES, endpoint=False)
    
    figures_dir = SCRIPT_DIR.parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Render multi-panel walking gait animation
    print(f"Rendering multi-panel walking gait animation ({NUM_FRAMES} frames)...")
    gait_frames = []
    for i, p in enumerate(thetas_all):
        if (i + 1) % 15 == 0 or i == 0:
            print(f"  Gait frame {i+1}/{NUM_FRAMES} (phi = {np.degrees(p):.1f} deg)...")
        frame = render_gait_frame(p, i, traj_data)
        gait_frames.append(frame)
    
    # Save as E02 and alias as E01
    out_gait_e02 = figures_dir / "E02_biped_walking_gait.gif"
    out_gait_e01 = figures_dir / "E01_biped_walking_gait.gif"
    gait_frames[0].save(
        out_gait_e02,
        save_all=True,
        append_images=gait_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"Saved: {out_gait_e02}")
    gait_frames[0].save(
        out_gait_e01,
        save_all=True,
        append_images=gait_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"Saved: {out_gait_e01}")
    
    # 2. Render focused schematic animation
    print(f"Rendering focused schematic animation ({NUM_FRAMES} frames)...")
    focus_frames = []
    for i, p in enumerate(thetas_all):
        if (i + 1) % 15 == 0 or i == 0:
            print(f"  Focus frame {i+1}/{NUM_FRAMES} (phi = {np.degrees(p):.1f} deg)...")
        frame = render_focus_frame(p, i, traj_data)
        focus_frames.append(frame)
    
    out_focus_e02 = figures_dir / "E02_biped_walking_focus.gif"
    out_focus_e01 = figures_dir / "E01_biped_walking_focus.gif"
    focus_frames[0].save(
        out_focus_e02,
        save_all=True,
        append_images=focus_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"Saved: {out_focus_e02}")
    focus_frames[0].save(
        out_focus_e01,
        save_all=True,
        append_images=focus_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"Saved: {out_focus_e01}")
    print("All animations generated successfully!")


if __name__ == "__main__":
    main()
