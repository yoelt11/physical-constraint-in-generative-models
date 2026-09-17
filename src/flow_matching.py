"""Minimal conditional flow-matching implementation, shared across toy
problems. Operates directly on ambient Cartesian coordinates -- there is
no learned latent space -- so a constraint function defined on the data
(h(x) = 0, g(x) <= 0) applies directly to the model's own input/output
space, with nothing to decode first.

A single dimension-agnostic MLP vector field is trained per toy problem
(dim=4 for the four-bar linkage's (x3,x4), dim=8 for the biped's
(knee_A, foot_A, knee_B, foot_B)) using the same code, per the project's
"shared code, separately trained per problem" design.

See experiments/E03_unconstrained_flow_matching for a worked example.
"""

import pickle

import jax
import jax.numpy as jnp
import optax


def init_params(rng, dim, hidden=(128, 128, 128)):
    sizes = [dim + 1] + list(hidden) + [dim]
    params = []
    for key, fan_in, fan_out in zip(jax.random.split(rng, len(sizes) - 1), sizes[:-1], sizes[1:]):
        w_key, _ = jax.random.split(key)
        W = jax.random.normal(w_key, (fan_in, fan_out)) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros((fan_out,))
        params.append((W, b))
    return params


def vector_field(params, x, t):
    """x: (..., dim), t: (...,) or scalar -> predicted velocity (..., dim)."""
    t = jnp.broadcast_to(t, x.shape[:-1])[..., None]
    h = jnp.concatenate([x, t], axis=-1)
    for W, b in params[:-1]:
        h = jax.nn.silu(h @ W + b)
    W, b = params[-1]
    return h @ W + b


def loss_fn(params, rng, x1, dim):
    """Conditional flow-matching loss: regress the constant velocity of
    the linear path between a noise sample and a data sample."""
    n = x1.shape[0]
    k0, kt = jax.random.split(rng)
    x0 = jax.random.normal(k0, (n, dim))
    t = jax.random.uniform(kt, (n,))
    xt = (1 - t)[:, None] * x0 + t[:, None] * x1
    target = x1 - x0
    pred = vector_field(params, xt, t)
    return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))


def train(rng, params, data, dim, steps=5000, batch_size=512, lr=1e-3, log_every=500):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, rng, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, rng, batch, dim)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    n = data.shape[0]
    losses = []
    for i in range(steps):
        rng, batch_key, loss_key = jax.random.split(rng, 3)
        idx = jax.random.randint(batch_key, (batch_size,), 0, n)
        params, opt_state, loss = step(params, opt_state, loss_key, data[idx])
        losses.append(float(loss))
        if log_every and i % log_every == 0:
            print(f"  step {i:6d}  loss {loss:.4f}")
    return params, losses


def sample(rng, params, dim, n, num_steps=100, return_trajectory=False):
    """Integrate dx/dt = v_theta(x, t) from t=0 (noise) to t=1 (data)
    with simple Euler steps."""
    x = jax.random.normal(rng, (n, dim))
    dt = 1.0 / num_steps
    trajectory = [x] if return_trajectory else None
    for step_i in range(num_steps):
        t = step_i * dt
        v = vector_field(params, x, jnp.full((n,), t))
        x = x + dt * v
        if return_trajectory:
            trajectory.append(x)
    if return_trajectory:
        return x, jnp.stack(trajectory)  # (num_steps + 1, n, dim)
    return x


def save_params(params, path):
    with open(path, "wb") as f:
        pickle.dump(jax.tree.map(lambda a: jax.device_get(a), params), f)


def load_params(path):
    with open(path, "rb") as f:
        return pickle.load(f)
