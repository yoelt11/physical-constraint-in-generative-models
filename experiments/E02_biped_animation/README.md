# E02: Two-Legged Biped Stick Figure Animation

This experiment synthesizes and animates the kinematic walking gait and configuration-space dynamics for the two-legged biped stick figure defined in `E01_problem_visualization`.

## Mechanism Geometry

- **Topology**: Tree topology (open kinematic chains), 4 DOF total (2 independent DOF per leg).
- **Hip Anchor**: $x_0 = (0, 0)$ (fixed pivot).
- **Leg A (Blue)**:
  - Knee $x_1$, foot $x_2$.
  - Link lengths: $\ell_{1A} = 1.0$ (thigh), $\ell_{2A} = 0.8$ (shin).
- **Leg B (Red)**:
  - Knee $x_3$, foot $x_4$.
  - Link lengths: $\ell_{1B} = 1.0$ (thigh), $\ell_{2B} = 0.8$ (shin).
- **Ground Level**: $y_{\rm ground} = -1.6660$ (matching the symmetric double-support pose in `E01_mechanism_schematic.png`).

## Gait Synthesis

- **Antiphase Coordination**: $\phi_A = \phi$, $\phi_B = (\phi + \pi) \pmod{2\pi}$.
- **Stance Phase ($\phi \in [0, \pi]$)**: Foot remains flush on the ground line ($y = y_{\rm ground}$) while moving backward relative to the hip, with compliant backward knee flexion.
- **Swing Phase ($\phi \in [\pi, 2\pi]$)**: Knee flexes backward to lift the foot $\approx 0.31$ units above ground level for clean forward stride clearance.
- **Equality Constraints**: All rigid link distances $\|x_{\rm knee} - x_0\| = \ell_1$ and $\|x_{\rm foot} - x_{\rm knee}\| = \ell_2$ are preserved with numerical precision ($\max \|h\| \sim 10^{-16}$).

## Generated Animations

1. **`E02_biped_walking_gait.gif`** (also aliased as `E01_biped_walking_gait.gif`):
   - **Panel (a)**: Full mechanism with fixed hip $x_0$, legs A & B, reachable annular workspace ($r \in [0.2, 1.8]$), knee orbit circle ($r = 1.0$), ground contact, and closed-loop foot trajectories.
   - **Panel (b)**: Configuration space limit cycles on the 2-torus $T^2 = (\theta_{\rm hip}, \theta_{\rm knee})$ with synchronized state markers.
   - **Panel (c)**: Foot ground clearance $y_{\rm foot}(\phi)$ showing stance contact and swing clearance over the gait cycle.

2. **`E02_biped_walking_focus.gif`** (also aliased as `E01_biped_walking_focus.gif`):
   - Single-panel animation matching `E01_mechanism_schematic.png` with complete joint/link annotations ($x_0, x_1, x_2, x_3, x_4, \ell_{1A}, \ell_{2A}, \ell_{1B}, \ell_{2B}$) and walking motion.
