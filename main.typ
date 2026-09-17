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
  title: [Toy Problems: Constrained Pose Synthesis for Small Mechanisms],
  keywords: (
    "flow matching",
    "constrained generation",
    "pose synthesis",
    "physics-informed generative models",
  ),
  abstract: [
    Three minimal geometric toy systems are studied for isolating
    constraint-enforcement mechanisms in flow matching. The first is a
    fixed four-body planar mechanism -- a four-bar linkage -- whose
    closed loop gives an analytically known, nonconvex feasible manifold
    with a discrete assembly-branch ambiguity. The second is a
    two-legged biped stick figure: an open tree of two independent limbs
    sharing a fixed hip, with no loop closure and hence no branch
    ambiguity, but more degrees of freedom concentrated within a single
    limb. The third reuses the biped's exact geometry and adds a
    gait-phase coupling between the two legs, concretely realizing a
    ground-contact constraint that is otherwise only described in the
    abstract. Together the three toy problems separate structural
    properties -- closed loop versus tree, branch multimodality versus
    none -- and constraint mechanisms -- geometric versus phase-dependent
    -- that any single toy would conflate. On the same coordinates,
    equality, inequality, orientation, semantic, and dynamical constraint
    families are defined, so that constraint-handling methods can be
    evaluated along two independent axes -- constraint violation and
    preservation of the true intrinsic distribution -- across
    structurally different mechanisms rather than just one.
  ],
  bibliography: none,
  header: [Constrained Pose Synthesis: Three Toy Mechanisms],
  appendix: none,
  // Anonymized: the template substitutes a placeholder author/affiliation
  // whenever accepted is false.
  accepted: false,
)

= Overview

Three minimal geometric systems are considered to isolate the effect of
constraint enforcement in flow-matching models, each built from a
handful of nodes $x_i in bb(R)^2$ connected by rigid links of fixed
length. In every case the mechanism itself -- which nodes are linked,
and the length of every link -- is fixed once and for all; the dataset
does not contain different mechanisms, only different valid *poses* of
a given one. The generative task is therefore pose synthesis for a
fixed mechanism, not shape generation across varying structures, and
the generative model always operates in Cartesian coordinates while the
valid data distribution is restricted by analytically known constraints.

The first two mechanisms differ in exactly the structural property that
matters most for this comparison. The four-bar linkage is a *closed
loop*, which has a single shared degree of freedom and a discrete
branch ambiguity. The two-legged biped is an *open tree* (two
independent limbs sharing a hip), which has no branch ambiguity but
several degrees of freedom concentrated within a single limb. The third
mechanism reuses the biped's exact structure and adds a gait-phase
coupling between the two legs, turning an otherwise-hypothetical
ground-contact constraint into a concretely realized one. This provides
a controlled setting in which constraint satisfaction can be evaluated
independently from distributional fidelity, across structurally
distinct mechanisms rather than just one.

= Toy Problem A: Four-Bar Linkage (Closed Loop)

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

// A plain #figure() is constrained to single-column width by the
// template's show rule, which is too small for this two-panel image to
// stay legible -- so this one spans the full page width instead, the
// same way LaTeX's figure* would, via place(..., scope: "parent").
#place(top + center, float: true, scope: "parent", clearance: 1.5em,
  block(width: 100%, breakable: false, {
    align(center, image("figures/E00_dataset_overview.png", width: 75%))
    v(0.1in, weak: true)
    align(center, block(width: 85%, text(size: 9pt)[
      _Figure 1._ Toy Problem A. The mechanism at a single labeled pose
      (left) and the reachable positions of the free nodes $x_3$, $x_4$
      across all valid poses (right). The mechanism itself -- topology
      and link lengths -- is fixed; only the pose ($theta$, $b$) varies
      across the dataset.
    ]))
  })
)

= Toy Problem B: Two-Legged Biped (Open Tree)

A second toy problem replaces the closed loop with a tree: a single
fixed hip $x_0$, with two independent open kinematic chains ("legs")
attached to it. Unlike the four-bar linkage, no link closes the loop
back to $x_0$, so there is no discrete assembly branch -- every choice
of joint angles is reachable and valid.

Each leg $ell in {A, B}$ has a knee $k_ell$ and a foot $f_ell$,

$
  k_ell = x_0 + l_1 (cos theta_(1 ell), sin theta_(1 ell)),
$

$
  f_ell = k_ell + l_2 (cos theta_(2 ell), sin theta_(2 ell)),
$

with shared thigh/shin lengths $l_1, l_2$ (both legs have identical
proportions, matching a real biped). In ambient Cartesian coordinates,
the generated state is $x = (k_A, f_A, k_B, f_B) in bb(R)^8$, subject to
the four equality constraints, for $ell in {A, B}$,

$
  h_(1 ell) (x) = norm(k_ell - x_0)^2 - l_1^2 = 0,
$

$
  h_(2 ell) (x) = norm(f_ell - k_ell)^2 - l_2^2 = 0.
$

Because each leg contributes two free joint angles and only two
constraints, four degrees of freedom survive in total (two per leg),
compared to the four-bar linkage's single, shared degree of freedom.
This has a visible geometric consequence: whereas the four-bar linkage's
individual free nodes were pinned to an *exact* 1-D circle (Figure 1), a
foot here sweeps a genuine 2-D annulus -- inner radius $|l_1 - l_2|$,
outer radius $l_1 + l_2$, centered on the hip -- since it depends on
*two* free angles rather than one (Figure 2). The tree topology is
therefore a natural complement to the closed loop: it removes the
branch ambiguity, but concentrates more degrees of freedom, and a
richer reachable set, within a single limb.

Ground contact and gait-phase coupling are realized concretely in Toy
Problem C, below. Foot-foot collision and anatomical knee-bend limits
remain open extensions (@sec-constraints), not yet implemented in
either toy problem's dataset generator; the figure below shows Toy
Problem B's static mechanism and its reachable region only.

#place(top + center, float: true, scope: "parent", clearance: 1.5em,
  block(width: 100%, breakable: false, {
    align(center, image("figures/E01_dataset_overview.png", width: 75%))
    v(0.1in, weak: true)
    align(center, block(width: 85%, text(size: 9pt)[
      _Figure 2._ Toy Problem B. The mechanism at a single labeled pose
      (left) and the reachable region of a knee and a foot (right). Both
      legs share the same link lengths; only the pose (four joint
      angles) varies across the dataset.
    ]))
  })
)

= Toy Problem C: Gait-Coupled Walking Biped

Toy Problem C reuses Toy Problem B's exact geometry -- the same hip
$x_0$, the same thigh/shin lengths $l_1, l_2$ -- and adds a single
shared gait-phase parameter $phi$ coupling the two legs,

$
  phi_A = phi, quad phi_B = (phi + pi) mod 2pi,
$

so that leg $B$ is always exactly half a cycle behind leg $A$. A fixed
ground level $y_"ground" approx -1.666$ then determines two regimes for
each leg. During *stance* ($phi_ell in [0, pi]$), its foot is held
exactly on the ground,

$
  y(f_ell) = y_"ground",
$

while during *swing* ($phi_ell in (pi, 2pi]$), the knee flexes further
to lift the foot clear,

$
  y(f_ell) > y_"ground".
$

This is a concrete realization of the semantic/conditional constraint
family (@sec-constraints): the discrete phase (stance vs. swing)
selects which equality or inequality applies to each foot, exactly as
$c$ does in $h(x, c) = 0$, $g(x, c) <= 0$. Unlike Toy Problems A and B,
this constraint is currently satisfied by a hand-designed kinematic law
-- the hip angle oscillates and the knee angle is solved, in closed
form, to keep the foot on the ground during stance -- implemented in
`experiments/E02_biped_animation/`, rather than by a general-purpose
dataset generator; it is not yet part of the `src/` pipeline alongside
Toy Problems A and B. An animated rendering is available in the
repository at `figures/E02_biped_walking_gait.gif`.

= Comparing the Three Toy Problems

Toy Problem C shares Toy Problem B's topology, generated state, and
degrees of freedom exactly -- the comparison below is therefore
structural, contrasting only Toy Problems A and B; what Toy Problem C
adds is a constraint, not a new structure, and is discussed in
@sec-constraints instead.

#place(top + center, float: true, scope: "parent", clearance: 1.5em,
  block(width: 100%, breakable: false, {
    align(center, table(
      columns: 3,
      stroke: 0.5pt,
      inset: 6pt,
      align: left,
      table.header([*Property*], [*A: four-bar linkage*], [*B: two-legged biped*]),
      [Topology], [closed loop (cycle)], [tree (two open chains)],
      [Generated state], [$x=(x_3,x_4) in bb(R)^4$], [$x in bb(R)^8$ (2 knees, 2 feet)],
      [Equality constraints], [3], [4],
      [Degrees of freedom], [1 ($theta$)], [4 (2 angles $times$ 2 legs)],
      [Discrete branch ambiguity], [yes ($b = plus.minus 1$)], [no],
      [Individual free-node locus],
        [exact 1-D circle],
        [knee: exact 1-D circle; foot: 2-D annulus],
      [Naturally active new constraint],
        [branch multimodality; orientation (signed area)],
        [gait phase; foot-foot collision; joint limits],
    ))
    v(0.1in, weak: true)
    align(center, block(width: 85%, text(size: 9pt)[
      _Table 1._ Neither toy problem alone exercises every constraint
      family well: closed-loop branch multimodality and orientation are
      specific to Toy A, while Toy B is the more natural host for
      node-node collision, and its gait-phase-coupled extension (Toy C)
      concretely realizes gait-phase conditioning (see
      @sec-constraints).
    ]))
  })
)

= Constraint Families <sec-constraints>

The same systems can be used to study several classes of constraints.

*Nonlinear equality constraints.* The linkage constraints above enforce
fixed pairwise distances and provide a simple example of generation on
an explicitly known manifold, realized directly by all three toy
problems.

*Inequality and collision constraints.* For nodes represented as disks of
radius $r_i$, collision avoidance is expressed as

$
  g_(i j) (x)
  = (r_i + r_j)^2 - norm(x_i - x_j)^2
  <= 0.
$

Obstacles can be introduced analogously by requiring generated nodes to
remain outside prescribed regions. In the four-bar linkage, only the two
node pairs *not* already joined by a rigid link -- $(x_1, x_3)$ and
$(x_2, x_4)$ -- have a separation that varies with the pose, so this
constraint is only ever non-trivial on those two pairs; every other pair
is pinned to a fixed distance by construction. The two-legged biped is a
better host for this family: the two feet are never rigidly linked to
each other, so a foot-foot collision inequality is active across the
whole pose space, not just on two special pairs.

*Orientation and topology constraints.* A signed-area constraint can be
used to distinguish valid from flipped or self-intersecting
configurations,

$
  A(x_1, x_2, x_3, x_4) > 0.
$

This provides a simple analogue of orientation and element-validity
constraints arising in mesh generation. The notion is specific to a
closed loop; the two-legged biped's tree topology has no analogous
signed-area constraint, though whether the two legs' segments cross each
other could be posed as a separate, collision-style constraint.

*Conditional or semantic constraints.* A context variable $c$ may modify
either the constraint parameters or the constraint type,

$
  h(x, c) = 0, quad g(x, c) <= 0.
$

For the four-bar linkage, an edge labeled as rigid may impose a
fixed-distance constraint, whereas a non-contact relation may impose a
minimum-distance inequality. Toy Problem C gives this family a
concretely realized instance: the gait phase $phi$ selects, independently
for each leg, whether its foot's height is pinned to the ground
($y = y_"ground"$, stance) or free to rise above it ($y > y_"ground"$,
swing).

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
This extension applies equally to either toy problem.

= Evaluation

On either toy problem, constrained flow-matching methods can be
evaluated along two independent axes: constraint satisfaction and
preservation of the target distribution. Constraint violation may be
quantified through residuals such as

$
  epsilon_c = expectation(norm(h(x))),
$

while distributional discrepancies can be measured in intrinsic
coordinates such as the joint angle(s), assembly branch (where one
exists), or other invariant geometric quantities. This distinction
prevents solutions that collapse onto a small subset of valid
configurations from being considered successful solely because the
constraints are satisfied.
