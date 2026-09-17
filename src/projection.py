"""Generic post-hoc projection onto a constraint manifold h(x) = 0, via
Gauss-Newton. Pure post-processing: takes points however they were
produced (here, an unconstrained flow-matching sample) and moves each
one to a nearby point satisfying the constraint, with no change to how
those points were generated.

h_fn(x) is any differentiable function R^dim -> R^m (m constraints);
its Jacobian is obtained automatically via jax.jacobian, not hand-
derived, so this is not specific to any one toy problem's constraints.
"""

import jax
import jax.numpy as jnp


def gauss_newton_project(x0, h_fn, max_iters=20, damping=1e-6):
    """x0: (n, dim) batch of points. Returns projected points of the same
    shape, each moved toward the manifold h_fn(x) = 0 by repeated
    minimum-norm Levenberg-Marquardt-damped Gauss-Newton corrections
    x <- x - J^T (J J^T + damping*I)^{-1} h(x), where J is m x dim
    (m = number of constraints, m < dim here). The damping term keeps
    the solve well-posed if J J^T is ever near-singular (rare, but
    observed for ~1 in 5000 unconstrained samples with damping=0)."""
    jac_fn = jax.jacobian(h_fn)

    def project_one(x):
        def body(x, _):
            r = h_fn(x)
            J = jac_fn(x)
            reg = damping * jnp.eye(J.shape[0])
            delta = J.T @ jnp.linalg.solve(J @ J.T + reg, r)
            return x - delta, None

        x_final, _ = jax.lax.scan(body, x, None, length=max_iters)
        return x_final

    return jax.vmap(project_one)(x0)
