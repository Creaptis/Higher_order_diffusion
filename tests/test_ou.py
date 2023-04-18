import jax
import jax.numpy as jnp
import os, sys
from samplers import euler_maruyama as em_sampler
from diffusion import transforms as diftils
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    data_shape = (16, 16, 3)
    B = 128
    x_init = jax.random.normal(rng, (B, *data_shape), dtype=jnp.float32)
    t_init = 0.0

    num_steps = 1000
    step_schedule = jnp.ones(num_steps) / num_steps

    f = lambda x, t: (50 - x) / 4
    g = lambda x, t: jnp.sqrt(2) * jnp.ones(x.shape[0])

    beta_t = lambda t: 20
    f, g = diftils.rescale_diffusion(f, g, beta_t)

    sampler = jax.jit(em_sampler.euler_maruyama, static_argnames=["f", "g"])

    diffusion_state = sampler(rng, x_init, t_init, step_schedule, f, g)
    final_samples_x = diffusion_state.x_t
    final_samples_x = final_samples_x.reshape(-1)
    print(jnp.mean(final_samples_x), jnp.std(final_samples_x))
