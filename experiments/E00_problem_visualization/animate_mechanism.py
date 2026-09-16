"""Animated visualization of the four-bar linkage mechanism.

Produces high-quality looping GIFs demonstrating:
1. E00_mechanism_side_by_side.gif: Side-by-side synchronized comparison of
   Assembly Branch b = +1 (left) and Branch b = -1 (right), showing the full
   kinematic loop, the coupler point xc tracing the nonconvex 1D feasible manifold,
   and the reachable loci of free nodes x3 and x4.
2. E00_mechanism_overlay.gif: Both branches simultaneously driven by the shared
   crank x2-x3 in a single coordinate frame.

Matches the geometric parameters in main.typ and visualize_problem.py:
  Fixed anchors: x1 = (0, 0), x2 = (l12, 0) with l12 = 1.0
  Links: l12 = 1.0, l23 = 0.6 (crank), l34 = 1.3 (coupler), l41 = 0.9 (rocker)
  Coupler point: xc = 0.5 * (x3 + x4)
"""

import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Import linkage solver and constants
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from visualize_problem import X1, X2, L12, L23, L34, L41, solve_linkage, BRANCH_COLORS

ANCHOR_COLOR = "#222222"
CRANK_COLOR = "#2a9d8f"      # teal/green for the input crank
COUPLER_COLOR_P = "#1f77b4"  # blue for branch +1
COUPLER_COLOR_M = "#d62728"  # red for branch -1
ROCKER_COLOR = "#e76f51"     # coral for the rocker link
COUPLER_PT_COLOR = "#9c27b0" # purple/magenta for coupler point

NUM_FRAMES = 120
FPS = 25


def compute_manifold_curves(num_points=1000):
    """Precompute the full coupler curves for background reference."""
    thetas = np.linspace(0, 2 * np.pi, num_points)
    curves = {+1: [], -1: []}
    x4_curves = {+1: [], -1: []}
    for t in thetas:
        x3, sols = solve_linkage(t)
        for b in (+1, -1):
            curves[b].append(0.5 * (x3 + sols[b]))
            x4_curves[b].append(sols[b])
    return {b: np.array(curves[b]) for b in (+1, -1)}, {b: np.array(x4_curves[b]) for b in (+1, -1)}


def draw_ground_hatch(ax, p, width=0.25, height=0.08, num_lines=4):
    """Draw small mechanical ground hatching underneath fixed anchors."""
    x, y = p
    ax.plot([x - width / 2, x + width / 2], [y - 0.04, y - 0.04], "-", color=ANCHOR_COLOR, lw=2.0, zorder=2)
    dx = width / num_lines
    for i in range(num_lines + 1):
        lx = x - width / 2 + i * dx
        ax.plot([lx, lx - 0.03], [y - 0.04, y - 0.04 - height], "-", color=ANCHOR_COLOR, lw=1.2, zorder=2)


def render_side_by_side_frame(theta, idx, thetas_all, full_coupler, full_x4):
    """Render a single frame for the side-by-side animation."""
    x3, solutions = solve_linkage(theta)
    
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    
    # Orbits
    circle_pts = np.linspace(0, 2 * np.pi, 250)
    x3_orbit = X2 + L23 * np.c_[np.cos(circle_pts), np.sin(circle_pts)]
    x4_orbit = X1 + L41 * np.c_[np.cos(circle_pts), np.sin(circle_pts)]
    
    # Active trace index
    trace_idx = idx + 1
    
    for i, b in enumerate([+1, -1]):
        ax = axes[i]
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.85)
        ax.set_ylim(-1.20, 1.20)
        ax.set_facecolor("#fafafa")
        ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
        
        # Draw orbits
        ax.plot(*x3_orbit.T, ":", color="#9e9e9e", lw=1.2, alpha=0.7, zorder=1)
        ax.plot(*x4_orbit.T, ":", color="#bdbdbd", lw=1.2, alpha=0.7, zorder=1)
        
        # Full static coupler curve (faint)
        cc_full = full_coupler[b]
        ax.plot(cc_full[:, 0], cc_full[:, 1], "--", color=BRANCH_COLORS[b],
                alpha=0.25, lw=1.5, zorder=1)
        
        # Trailing path of coupler point up to current frame
        trail_pts = [0.5 * (solve_linkage(t)[0] + solve_linkage(t)[1][b]) for t in thetas_all[:trace_idx]]
        if len(trail_pts) > 1:
            trail_pts = np.array(trail_pts)
            ax.plot(trail_pts[:, 0], trail_pts[:, 1], "-", color=BRANCH_COLORS[b],
                    lw=2.5, alpha=0.85, zorder=2)
        
        # Trailing arc of x4
        x4_trail = [solve_linkage(t)[1][b] for t in thetas_all[:trace_idx]]
        if len(x4_trail) > 1:
            x4_trail = np.array(x4_trail)
            ax.plot(x4_trail[:, 0], x4_trail[:, 1], "-", color=ROCKER_COLOR,
                    lw=1.8, alpha=0.6, zorder=2)
        
        x4 = solutions[b]
        xc = 0.5 * (x3 + x4)
        
        # Ground hatching
        draw_ground_hatch(ax, X1)
        draw_ground_hatch(ax, X2)
        
        # Links
        # Link 12 (ground)
        ax.plot([X1[0], X2[0]], [X1[1], X2[1]], "-", color="#333333", lw=4.5,
                solid_capstyle="round", zorder=3)
        # Link 23 (crank)
        ax.plot([X2[0], x3[0]], [X2[1], x3[1]], "-", color=CRANK_COLOR, lw=3.8,
                solid_capstyle="round", zorder=4)
        # Link 34 (coupler)
        ax.plot([x3[0], x4[0]], [x3[1], x4[1]], "-", color=BRANCH_COLORS[b], lw=3.8,
                solid_capstyle="round", zorder=4)
        # Link 41 (rocker)
        ax.plot([x4[0], X1[0]], [x4[1], X1[1]], "-", color=ROCKER_COLOR, lw=3.8,
                solid_capstyle="round", zorder=4)
        
        # Nodes
        # Fixed anchors (squares)
        ax.scatter(*X1, color=ANCHOR_COLOR, marker="s", s=130, zorder=6, label=r"Anchor $x_1=(0,0)$")
        ax.scatter(*X2, color=ANCHOR_COLOR, marker="s", s=130, zorder=6, label=r"Anchor $x_2=(\ell_{12},0)$")
        
        # Moving joints (circles)
        ax.scatter(*x3, color=CRANK_COLOR, marker="o", s=120, edgecolors="white",
                   linewidths=1.5, zorder=7, label=r"Crank node $x_3$")
        ax.scatter(*x4, color=BRANCH_COLORS[b], marker="o", s=120, edgecolors="white",
                   linewidths=1.5, zorder=7, label=r"Rocker node $x_4$")
        
        # Coupler point (diamond)
        ax.scatter(*xc, color=COUPLER_PT_COLOR, marker="D", s=110, edgecolors="white",
                   linewidths=1.5, zorder=8, label=r"Coupler point $x_c$ (manifold)")
        
        # Text labels near nodes
        ax.annotate(r"$x_1$", X1, textcoords="offset points", xytext=(-16, -18),
                    fontsize=12, fontweight="bold", color=ANCHOR_COLOR)
        ax.annotate(r"$x_2$", X2, textcoords="offset points", xytext=(8, -18),
                    fontsize=12, fontweight="bold", color=ANCHOR_COLOR)
        ax.annotate(r"$x_3$", x3, textcoords="offset points", xytext=(8, 8),
                    fontsize=12, fontweight="bold", color=CRANK_COLOR)
        y_off_4 = 10 if b == 1 else -18
        ax.annotate(r"$x_4$", x4, textcoords="offset points", xytext=(-18, y_off_4),
                    fontsize=12, fontweight="bold", color=BRANCH_COLORS[b])
        y_off_c = 10 if b == 1 else -18
        ax.annotate(r"$x_c$", xc, textcoords="offset points", xytext=(10, y_off_c),
                    fontsize=11, fontweight="bold", color=COUPLER_PT_COLOR)
        
        # Crank angle arc at x2
        arc_r = 0.22
        ang_deg = np.degrees(theta) % 360
        arc_angles = np.linspace(0, theta, max(int(ang_deg / 3), 4))
        arc_pts = X2 + arc_r * np.c_[np.cos(arc_angles), np.sin(arc_angles)]
        ax.plot(*arc_pts.T, "-", color="#555555", lw=1.2, zorder=5)
        
        # Branch title & residuals
        h1 = (np.linalg.norm(x3 - X2) - L23)
        h2 = (np.linalg.norm(x4 - x3) - L34)
        h3 = (np.linalg.norm(x4 - X1) - L41)
        res_norm = np.sqrt(h1**2 + h2**2 + h3**2)
        
        branch_sign = "+1" if b == 1 else "-1"
        ax.set_title(f"Assembly Branch $b = {branch_sign}$", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("$x$", fontsize=11)
        ax.set_ylabel("$y$", fontsize=11)
        
        # Info box inside axes
        info_text = (
            rf"$\theta = {ang_deg:5.1f}^\circ$" + "\n"
            rf"$\|h(x)\| = {res_norm:.1e}$"
        )
        ax.text(0.04, 0.96, info_text, transform=ax.transAxes, fontsize=9.5,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))
        
        if i == 0:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
        else:
            # Legend with link lengths
            link_text = (
                r"$\ell_{12} = 1.00$ (ground)" + "\n"
                r"$\ell_{23} = 0.60$ (crank)" + "\n"
                r"$\ell_{34} = 1.30$ (coupler)" + "\n"
                r"$\ell_{41} = 0.90$ (rocker)"
            )
            ax.text(0.96, 0.04, link_text, transform=ax.transAxes, fontsize=8.5,
                    verticalalignment="bottom", horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.95))
    
    fig.suptitle(r"Four-Bar Linkage Pose Synthesis: Valid Manifold & Dual Assembly Branches",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Render to PIL Image
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba)
    plt.close(fig)
    return img


def render_overlay_frame(theta, idx, thetas_all, full_coupler, full_x4):
    """Render a single frame for the combined single-panel animation."""
    x3, solutions = solve_linkage(theta)
    
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.85)
    ax.set_ylim(-1.20, 1.20)
    ax.set_facecolor("#fafafa")
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    
    # Orbits
    circle_pts = np.linspace(0, 2 * np.pi, 250)
    x3_orbit = X2 + L23 * np.c_[np.cos(circle_pts), np.sin(circle_pts)]
    x4_orbit = X1 + L41 * np.c_[np.cos(circle_pts), np.sin(circle_pts)]
    ax.plot(*x3_orbit.T, ":", color="#9e9e9e", lw=1.2, alpha=0.7, zorder=1)
    ax.plot(*x4_orbit.T, ":", color="#bdbdbd", lw=1.2, alpha=0.7, zorder=1)
    
    # Full coupler curves
    for b in (+1, -1):
        cc_full = full_coupler[b]
        ax.plot(cc_full[:, 0], cc_full[:, 1], "--", color=BRANCH_COLORS[b],
                alpha=0.3, lw=1.5, zorder=1)
    
    # Trailing paths
    trace_idx = idx + 1
    for b in (+1, -1):
        trail = [0.5 * (solve_linkage(t)[0] + solve_linkage(t)[1][b]) for t in thetas_all[:trace_idx]]
        if len(trail) > 1:
            trail = np.array(trail)
            ax.plot(trail[:, 0], trail[:, 1], "-", color=BRANCH_COLORS[b],
                    lw=2.5, alpha=0.85, zorder=2)
    
    # Anchors and ground link
    draw_ground_hatch(ax, X1)
    draw_ground_hatch(ax, X2)
    ax.plot([X1[0], X2[0]], [X1[1], X2[1]], "-", color="#333333", lw=4.5,
            solid_capstyle="round", zorder=3, label=r"Ground link $\ell_{12}=1.0$")
    
    # Shared crank
    ax.plot([X2[0], x3[0]], [X2[1], x3[1]], "-", color=CRANK_COLOR, lw=4.0,
            solid_capstyle="round", zorder=4, label=r"Input crank $\ell_{23}=0.6$")
    
    # Both branches
    for b in (+1, -1):
        x4 = solutions[b]
        xc = 0.5 * (x3 + x4)
        b_name = "b = +1" if b == 1 else "b = -1"
        
        # Coupler
        ax.plot([x3[0], x4[0]], [x3[1], x4[1]], "-", color=BRANCH_COLORS[b], lw=3.2,
                solid_capstyle="round", zorder=4, label=f"Coupler link $\\ell_{{34}}$ ({b_name})")
        # Rocker
        ax.plot([x4[0], X1[0]], [x4[1], X1[1]], "-", color=ROCKER_COLOR if b == 1 else "#e76f51",
                linestyle="solid", lw=3.0, alpha=0.9, solid_capstyle="round", zorder=4)
        
        # Moving node x4
        ax.scatter(*x4, color=BRANCH_COLORS[b], marker="o", s=110, edgecolors="white",
                   linewidths=1.5, zorder=7)
        # Coupler point xc
        ax.scatter(*xc, color=COUPLER_PT_COLOR, marker="D", s=100, edgecolors="white",
                   linewidths=1.5, zorder=8)
        
        y_off = 10 if b == 1 else -18
        ax.annotate(rf"$x_4^{{({'+1' if b==1 else '-1'})}}$", x4, textcoords="offset points",
                    xytext=(-24, y_off), fontsize=11, fontweight="bold", color=BRANCH_COLORS[b])
        ax.annotate(rf"$x_c^{{({'+1' if b==1 else '-1'})}}$", xc, textcoords="offset points",
                    xytext=(10, y_off), fontsize=10, fontweight="bold", color=COUPLER_PT_COLOR)
    
    # Shared anchors and crank node
    ax.scatter(*X1, color=ANCHOR_COLOR, marker="s", s=130, zorder=6)
    ax.scatter(*X2, color=ANCHOR_COLOR, marker="s", s=130, zorder=6)
    ax.scatter(*x3, color=CRANK_COLOR, marker="o", s=120, edgecolors="white",
               linewidths=1.5, zorder=7)
    
    ax.annotate(r"$x_1$", X1, textcoords="offset points", xytext=(-16, -18),
                fontsize=12, fontweight="bold", color=ANCHOR_COLOR)
    ax.annotate(r"$x_2$", X2, textcoords="offset points", xytext=(8, -18),
                fontsize=12, fontweight="bold", color=ANCHOR_COLOR)
    ax.annotate(r"$x_3$", x3, textcoords="offset points", xytext=(8, 8),
                fontsize=12, fontweight="bold", color=CRANK_COLOR)
    
    # Angle arc
    arc_r = 0.22
    ang_deg = np.degrees(theta) % 360
    arc_angles = np.linspace(0, theta, max(int(ang_deg / 3), 4))
    arc_pts = X2 + arc_r * np.c_[np.cos(arc_angles), np.sin(arc_angles)]
    ax.plot(*arc_pts.T, "-", color="#555555", lw=1.2, zorder=5)
    
    ax.set_title(rf"Four-Bar Mechanism: Dual Branches ($b = \pm 1$, $\theta = {ang_deg:5.1f}^\circ$)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$y$", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(rgba)
    plt.close(fig)
    return img


def main():
    print(f"Precomputing manifold curves...")
    full_coupler, full_x4 = compute_manifold_curves(1000)
    thetas_all = np.linspace(0, 2 * np.pi, NUM_FRAMES, endpoint=False)
    
    figures_dir = SCRIPT_DIR.parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Render Side-by-side animation
    print(f"Rendering side-by-side animation ({NUM_FRAMES} frames)...")
    side_frames = []
    for i, t in enumerate(thetas_all):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Frame {i+1}/{NUM_FRAMES} (theta = {np.degrees(t):.1f} deg)...")
        frame = render_side_by_side_frame(t, i, thetas_all, full_coupler, full_x4)
        side_frames.append(frame)
    
    out_side = figures_dir / "E00_mechanism_side_by_side.gif"
    side_frames[0].save(
        out_side,
        save_all=True,
        append_images=side_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"Saved side-by-side GIF to: {out_side}")
    
    # Also save as primary E00_four_bar_mechanism.gif
    out_primary = figures_dir / "E00_four_bar_mechanism.gif"
    side_frames[0].save(
        out_primary,
        save_all=True,
        append_images=side_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"Saved primary GIF to: {out_primary}")
    
    # 2. Render Overlay animation
    print(f"Rendering overlay animation ({NUM_FRAMES} frames)...")
    overlay_frames = []
    for i, t in enumerate(thetas_all):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Frame {i+1}/{NUM_FRAMES} (theta = {np.degrees(t):.1f} deg)...")
        frame = render_overlay_frame(t, i, thetas_all, full_coupler, full_x4)
        overlay_frames.append(frame)
    
    out_overlay = figures_dir / "E00_mechanism_overlay.gif"
    overlay_frames[0].save(
        out_overlay,
        save_all=True,
        append_images=overlay_frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"Saved overlay GIF to: {out_overlay}")


if __name__ == "__main__":
    main()
