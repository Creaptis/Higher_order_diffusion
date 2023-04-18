import jax.numpy as jnp
import jax
import flax

from typing import Callable
from data_types import RNG, Array

import diffusion.base as diffusion_lib
import diffusion.beta_schedule as schedule_lib

import utils.math as ops


def compute_ou_moments(mean_T, sigma_T, init_x, t):
    sigma_inv = 1 / sigma_T

    x_weight = jnp.exp(-0.5 * ops.batch_mul(sigma_inv, t, in_axes=(None, 0)))
    mean_t = ops.batch_mul(x_weight, init_x) + ops.batch_mul(
        1 - x_weight, mean_T, in_axes=(0, None)
    )

    sigma_weight = jnp.sqrt(
        1 - jnp.exp(-ops.batch_mul(sigma_inv, t, in_axes=(None, 0)))
    )

    sigma_t = ops.batch_mul(sigma_T, sigma_weight, in_axes=(None, 0))
    return mean_t, sigma_t


def get_ornstein_uhlenbeck_sampler(
    mean_T: Array, sigma_T: Array, beta_schedule: schedule_lib.BetaSchedule
):
    def sampler(
        diffusion_state: diffusion_lib.DiffusionState,
        t: Array,
    ) -> diffusion_lib.DiffusionState:
        integral_beta_t = beta_schedule.integral_beta_t(t)
        integral_beta_t = integral_beta_t - beta_schedule.integral_beta_t(
            diffusion_state.t
        )
        mean_t, sigma_t = compute_ou_moments(
            mean_T=mean_T,
            sigma_T=sigma_T,
            init_x=diffusion_state.x_t,
            t=integral_beta_t,
        )

        rng, step_rng = jax.random.split(diffusion_state.rng)
        noise = jax.random.normal(step_rng, diffusion_state.x_t.shape)
        x_next = mean_t + ops.batch_mul(sigma_t, noise)

        return diffusion_state.replace(
            rng=rng, x_t=x_next, mean_t=mean_t, t=t, noise=noise
        )

    return sampler


if __name__ == "__main__":
    beta_min = 0.001
    beta_max = 30.0
    t_0 = 0.0
    T_max = 1.0
    B = 100
    data_shape = (2,)
    mean_T = jnp.array([2.0])
    sigma_T = jnp.array([0.1])

    x_init = jnp.zeros((B, *data_shape))
    t = jnp.ones((B,)) * T_max
    rng = jax.random.PRNGKey(0)

    diffusion_state = diffusion_lib.init_diffusion_state(x_init=x_init, rng=rng)

    beta_schedule = beta_schedule = schedule_lib.LinearSchedule(
        beta_0=beta_min, beta_T=beta_max, t_0=t_0, T=T_max
    )
    sampler = get_ornstein_uhlenbeck_sampler(
        mean_T=mean_T,
        sigma_T=sigma_T,
        beta_schedule=beta_schedule,
    )

    forward_state = sampler(diffusion_state, t=t)
    print(jnp.mean(forward_state.x_t, axis=0), jnp.std(forward_state.x_t, axis=0))
    mean_t, sigma_t = compute_ou_moments(mean_T, sigma_T, init_x=x_init, t=t)
    print(mean_t[0], sigma_t[0])
