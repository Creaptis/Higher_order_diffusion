import jax.numpy as jnp
import utils.math as ops
import diffusion.transforms as transform_lib


def get_ornstein_uhlenbeck_diffusion(mean_T, sigma_T, beta_schedule=None):
    """Return the drift function of the Ornstein-Uhlenbeck process.

    Args:
        mean_T: the mean of stationary distribution.
        sigma_T: the standard deviation of stationary distribution. Only diagonal sigma_T is supported.
    """

    def drift_fn(x, t):
        adj_mean = ops.batch_add(mean_T, -x, in_axes=(None, 0))
        return ops.batch_mul(adj_mean, 1 / sigma_T**2, in_axes=(0, None))

    diffusion_scale_fn = lambda x, t: jnp.sqrt(2) * jnp.ones(x.shape[0])
    if beta_schedule is not None:
        drift_fn, diffusion_scale_fn = transform_lib.rescale_diffusion(
            beta_fn=beta_schedule.beta_t,
            drift_fn=drift_fn,
            diffusion_scale_fn=diffusion_scale_fn,
        )
    return drift_fn, diffusion_scale_fn


def get_forward_brownian_bridge_drift(x_T, T):
    """Return the drift function of the Brownian bridge.

    Args:
        x_T: the terminal state.
        T: the terminal time.
    """

    def drift_fn(x, t):
        return ops.batch_mul((x_T - x), 1 / (T - t))

    return drift_fn


if __name__ == "__main__":
    from samplers import euler_maruyama
    import diffusion.base as diffusion_lib
    import diffusion.beta_schedule as schedule_lib
    import jax

    beta_max = 5.0
    beta_min = 1e-4

    T_max = 1.0
    t_0 = 0

    beta_schedule = schedule_lib.LinearSchedule(
        beta_0=beta_min, beta_T=beta_max, t_0=t_0, T=T_max
    )
    N = 512
    mean_T = jnp.zeros((2,))
    sigma_T = jnp.ones((2,)) * 0.5
    drift_fn, scale_fn = get_ornstein_uhlenbeck_diffusion(
        mean_T, sigma_T, beta_schedule=beta_schedule
    )

    x_init = jnp.zeros((N, 2))
    t_init = jnp.zeros((N,))

    rng = jax.random.PRNGKey(0)
    diffusion_state = diffusion_lib.init_diffusion_state(
        x_init=x_init, t_init=t_init, rng=rng
    )

    num_steps = 1_000
    step_schedule = jnp.ones(num_steps) * T_max / num_steps

    print(beta_schedule.beta_t(0))
    print(beta_schedule.beta_t(jnp.cumsum(step_schedule)))

    diffusion_state = euler_maruyama(
        diffusion_state=diffusion_state,
        step_schedule=step_schedule,
        drift_fn=drift_fn,
        diffusion_scale_fn=scale_fn,
        return_trajectory=False,
    )

    print(jax.tree_map(lambda x: x.shape, diffusion_state))
    print(jnp.mean(diffusion_state.x_t, axis=0))
    print(jnp.std(diffusion_state.x_t, axis=0))
