import jax
import jax.numpy as jnp

from typing import Union
from jaxtyping import Array, Float, Int

import functools
import utils.math as ops
from data_types import RNG, DiffusionFn, DriftFn
import diffusion.base as diffusion_lib
import samplers.base as sampler_lib


def _euler_maruyama_step(
    rng: jax.random.KeyArray,
    x: Array,
    drift: Array,
    diffusion_scale: Array,
    step_size: float,
):
    """Simulate one step of the SDE using the Euler-Maruyama method.

    Args:
        rng: a JAX PRNGKey.
        x: the current state.
        drift: the drift of the SDE.
        diffusion_scale: the diffusion scale of the SDE.
        step_size: the time step size.
    """
    noise = jax.random.normal(rng, drift.shape)  # [Array, 'b *d']
    x_mean = x + step_size * drift  # [Array, 'b d']
    x = x_mean + jnp.sqrt(step_size) * ops.batch_mul(
        diffusion_scale, noise
    )  # [Array, 'b d*']
    return x, x_mean, noise


def euler_maruyama_step(
    diffusion_state: diffusion_lib.DiffusionState,
    step_size: float,
    drift_fn: DriftFn,
    diffusion_scale_fn: DiffusionFn,
) -> diffusion_lib.DiffusionState:
    """Simulate one step of the SDE using the Euler-Maruyama method.

    Args:
        rng: a JAX PRNGKey.
        x: the current state.
        t: the current time.
        step_size: the time step size.
        drift_fn: a function that takes in the state and time and returns the drift.
        diffusion_scale_fn: a function that takes in the state and time and returns the diffusion.
    """
    rng = diffusion_state.rng
    rng, step_rng = jax.random.split(rng)

    drift = drift_fn(diffusion_state.x_t, diffusion_state.t)  # [Array, 'b *d']
    diffusion_scale = diffusion_scale_fn(
        diffusion_state.x_t, diffusion_state.t
    )  # [Array, 'b *d']

    x, x_mean, noise = _euler_maruyama_step(
        step_rng, diffusion_state.x_t, drift, diffusion_scale, step_size
    )
    t = diffusion_state.t + step_size

    return diffusion_state.update(x_t=x, mean_t=x_mean, t=t, rng=rng, noise=noise)


def euler_maruyama_to_time_t(
    diffusion_state: diffusion_lib.DiffusionState,
    t: Array,
    n_steps: int,
    drift_fn: DriftFn,
    diffusion_scale_fn: DiffusionFn,
    return_trajectory: bool = False,
) -> diffusion_lib.DiffusionState:
    """Simulate multiple step of the SDE using the Euler-Maruyama method.

    Args:
    diffusion_state: the initial state.
    n_steps: the number of steps to simulate.
    drift_fn: a function that takes in the state and time and returns the drift.
    diffusion_scale_fn: a function that takes in the state and time and returns the diffusion.
    return_trajectory: whether to store the trajectory.

    Returns:
    diffusion_state: the final state.
    """

    step_schedule = jnp.ones((n_steps,)) * (t - diffusion_state.t) / n_steps
    return euler_maruyama(
        diffusion_state=diffusion_state,
        step_schedule=step_schedule,
        drift_fn=drift_fn,
        diffusion_scale_fn=diffusion_scale_fn,
        return_trajectory=return_trajectory,
    )


def euler_maruyama(
    diffusion_state: diffusion_lib.DiffusionState,
    step_schedule: Float[Array, "B n_steps"],
    drift_fn: DriftFn,
    diffusion_scale_fn: DiffusionFn,
    return_trajectory: bool = False,
):
    """Simulate multiple steps of the SDE using the Euler-Maruyama method.

    Args:
        rng: a JAX PRNGKey.
        init_state: the initial state.
        step_schedule: the time step.
        f: a function that takes in the state and time and returns the drift.
        g: a function that takes in the state and time and returns the diffusion.
        return_trajectory: whether to store the trajectory.
    """

    step_fn = functools.partial(
        euler_maruyama_step, drift_fn=drift_fn, diffusion_scale_fn=diffusion_scale_fn
    )

    return sampler_lib.simulate_diffusion(
        diffusion_state=diffusion_state,
        step_fn=step_fn,
        transition_inputs=step_schedule,
        return_trajectory=return_trajectory,
    )


if __name__ == "__main__":
    B = 10
    d = 2
    x_init = jnp.ones((B, d)) * 0.5
    t = jnp.zeros((B,))
    f = lambda x, t: jnp.ones(x.shape)
    g = lambda x, t: jnp.zeros(x.shape[0])
    dt = 0.1
    n_steps = 20
    rng = jax.random.PRNGKey(0)
    step_schedule = jnp.ones((n_steps,)) * dt

    euler_maruyama = jax.jit(
        euler_maruyama,
        static_argnames=("drift_fn", "diffusion_scale_fn", "return_trajectory"),
    )

    diffusion_state = diffusion_lib.init_diffusion_state(
        x_init=x_init, t_init=t, rng=rng
    )
    diffusion_state = euler_maruyama(
        diffusion_state=diffusion_state,
        step_schedule=step_schedule,
        drift_fn=f,
        diffusion_scale_fn=g,
    )
    print(jax.tree_map(lambda x: x.shape, diffusion_state))
    print(diffusion_state.t)

    diffusion_state = euler_maruyama(
        diffusion_state=diffusion_state,
        step_schedule=step_schedule,
        drift_fn=f,
        diffusion_scale_fn=g,
        return_trajectory=True,
    )

    print(jax.tree_map(lambda x: x.shape, diffusion_state))

    init_state = diffusion_lib.DiffusionState(
        rng=rng,
        x_t=x_init,
        mean_t=x_init,
        t=t,
        noise=jnp.zeros_like(x_init),
        x_0=x_init,
    )
    step_fn = functools.partial(euler_maruyama_step, drift_fn=f, diffusion_scale_fn=g)

    store_state_time_index = jax.vmap(
        lambda rng_step: jax.random.choice(
            rng_step,
            n_steps + 1,
            (15,),
            replace=False,
        ),
        in_axes=0,
    )(jax.random.split(rng, B))
    store_state_time_index = jnp.sort(store_state_time_index, axis=1)

    stored_states = sampler_lib.partial_diffusion_trajectory(
        init_state,
        step_fn=step_fn,
        transition_inputs=step_schedule,
        store_state_keys=["x_t", "mean_t", "t", "noise"],
        store_state_time_index=store_state_time_index,
    )

    k = 0
    print(jax.tree_map(lambda x: x.shape, stored_states))
    print(store_state_time_index[k])
    print("stored_states", stored_states["x_t"][:, k])
    # print(full_trajectory["x_t"][:, k])

    # print(
    #     "stored_states",
    #     stored_states["x_t"][:, k]
    #     - full_trajectory["x_t"][store_state_time_index[k] - 1, k],
    # )
