// Note: imported from a vendored, patched copy (vendor/lucky-icml), not the
// @preview package directly -- lucky-icml 0.7.0 targets Typst 0.12 and
// crashes on Typst 0.14+ (type() now returns a `type` value, not a string,
// breaking an unconditional string concatenation in its affiliation
// formatting). See vendor/lucky-icml/icml2024.typ for the one-line fix.
#import "vendor/lucky-icml/icml2025.typ": icml2025

// Custom math helpers (not provided by the template).
#let norm(x) = $ lr(bar.v.double #x bar.v.double) $
#let expectation(x) = $ upright(E)[#x] $

#show: icml2025.with(
  title: [Toy Problem: Constrained Four-Node System],
  keywords: (
    "flow matching",
    "constrained generation",
    "physics-informed generative models",
  ),
  abstract: [
    A minimal geometric toy system is studied for isolating
    constraint-enforcement mechanisms in flow matching. A four-node planar
    linkage, parameterized by an intrinsic joint angle and a discrete
    assembly branch, provides an analytically known, nonconvex feasible
    manifold together with a controllable, multimodal target distribution.
    On the same coordinates, equality, inequality, orientation, semantic,
    and dynamical constraint families are defined, so that a single
    generative model and dataset can be reused across constraint types.
    This allows constraint-handling methods to be evaluated along two
    independent axes -- constraint violation and preservation of the true
    intrinsic distribution -- rather than validity rate alone.
  ],
  bibliography: none,
  header: [Constrained Four-Node System],
  appendix: none,
  // Anonymized: the template substitutes a placeholder author/affiliation
  // whenever accepted is false.
  accepted: false,
)

= Overview

A minimal geometric system of four nodes, $x_i in bb(R)^2$, is considered
to isolate the effect of constraint enforcement in flow-matching models.
The generative model operates in Cartesian coordinates, while the valid
data distribution is restricted by analytically known constraints. This
provides a controlled setting in which constraint satisfaction can be
evaluated independently from distributional fidelity.

= Four-bar Linkage

The nodes are connected cyclically as $1 arrow 2 arrow 3 arrow 4 arrow 1$.
To remove global translation and rotation, the first two nodes are fixed
as

$
  x_1 = (0, 0), quad
  x_2 = (l_12, 0).
$

The generated state is therefore $x = (x_3, x_4) in bb(R)^4$. A valid
configuration satisfies the nonlinear equality constraints

$
  h_1 (x) = norm(x_3 - x_2)^2 - l_23^2 = 0,
$

$
  h_2 (x) = norm(x_4 - x_3)^2 - l_34^2 = 0,
$

$
  h_3 (x) = norm(x_4 - x_1)^2 - l_41^2 = 0.
$

These constraints define a low-dimensional nonlinear manifold in the
ambient generative space. Valid configurations can be parameterized by an
intrinsic joint angle $theta$ and, in general, by two assembly branches. A
multimodal distribution over these intrinsic variables can be selected
such that satisfying $h(x) = 0$ alone is insufficient to recover the
target distribution.

= Constraint Families

The same system can be used to study several classes of constraints.

*Nonlinear equality constraints.* The linkage constraints above enforce
fixed pairwise distances and provide a simple example of generation on an
explicitly known manifold.

*Inequality and collision constraints.* For nodes represented as disks of
radius $r_i$, collision avoidance is expressed as

$
  g_(i j) (x)
  = (r_i + r_j)^2 - norm(x_i - x_j)^2
  <= 0.
$

Obstacles can be introduced analogously by requiring generated nodes to
remain outside prescribed regions.

*Orientation and topology constraints.* A signed-area constraint can be
used to distinguish valid from flipped or self-intersecting
configurations,

$
  A(x_1, x_2, x_3, x_4) > 0.
$

This provides a simple analogue of orientation and element-validity
constraints arising in mesh generation.

*Conditional or semantic constraints.* A context variable $c$ may modify
either the constraint parameters or the constraint type,

$
  h(x, c) = 0, quad g(x, c) <= 0.
$

For example, an edge labeled as rigid may impose a fixed-distance
constraint, whereas a non-contact relation may impose a minimum-distance
inequality. This setting enables the study of constraints whose form
depends on discrete semantic information.

*Dynamical constraints.* By augmenting each node with a velocity $v_i$,
admissible velocities for a holonomically constrained system satisfy

$
  J_h (x) v = 0,
$

where $J_h$ denotes the Jacobian of the geometric constraints. Additional
physical invariants, such as a prescribed energy,

$
  E(x, v) = E_0,
$

can be imposed to study differential and conservation-based constraints.

= Evaluation

Constrained flow-matching methods can be evaluated along two independent
axes: constraint satisfaction and preservation of the target
distribution. Constraint violation may be quantified through residuals
such as

$
  epsilon_c = expectation(norm(h(x))),
$

while distributional discrepancies can be measured in intrinsic
coordinates such as the joint angle, assembly branch, or other invariant
geometric quantities. This distinction prevents solutions that collapse
onto a small subset of valid configurations from being considered
successful solely because the constraints are satisfied.
