import jax.numpy as jnp


def fixed_schedule(n_steps, t_init=0.0, t_end=1.0):
    """Returns a fixed schedule of step sizes for diffusion."""
    return jnp.ones((n_steps,)) * (t_end - t_init) / n_steps


def linear_schedule(n_steps, t_init=0.0, t_end=1.0):
    """Returns a linear schedule of step sizes for diffusion."""
    return jnp.linspace(t_init, t_end, n_steps)


def cosine_schedule(n_steps, t_init=0.0, t_end=1.0):
    """Returns a cosine schedule of step sizes for diffusion."""
    return (1 - jnp.cos(jnp.linspace(0, jnp.pi, n_steps))) * (t_end - t_init) / 2
