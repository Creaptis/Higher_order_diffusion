from utils import math as ops
import jax.numpy as jnp
from typing import Iterable


def rescale_diffusion(drift_fn, diffusion_scale_fn, beta_fn):
    """Rescale the diffusion.

    Args:
        drift_fn: a function that takes in the state and time and returns the drift.
        diffusion_scale_fn: a function that takes in the state and time and returns the diffusion.
        beta_fn: a function that takes in the time and returns the rescaling factor.
    """

    def rescaled_drift_fn(x, t):
        return ops.batch_mul(drift_fn(x, t), beta_fn(t))

    def rescaled_diffusion_scale_fn(x, t):
        return ops.batch_mul(diffusion_scale_fn(x, t), jnp.sqrt(beta_fn(t)))

    return rescaled_drift_fn, rescaled_diffusion_scale_fn


def _reverse_time(func, T_max=1, argnum=1):
    """Reverse the time argument of a function.

    Args:
        func: a function.
        T_max: the maximum time.
        argnum: the index of the time argument.
    """

    def reversed_func(*args):
        args = list(args)
        args[argnum] = T_max - args[argnum]
        return func(*args)

    return reversed_func


def reverse_time_argument(funcs, T_max=1, argnum=1):
    """Reverse the time argument of a function or a list or tuple of functions.

    func(x, t, ...) -> func(x, T - t, ...)

    Args:
        funcs: a function, list or tuple of functions.
        T_max: the maximum time.
    """
    if isinstance(funcs, Iterable):
        iterable_type = type(funcs)
        return iterable_type(
            _reverse_time(f, T_max=T_max, argnum=argnum) for f in funcs
        )
    else:
        return _reverse_time(funcs, T_max=T_max, argnum=argnum)


def time_reversal(
    forward_drift_fn,
    forward_diffusion_scale_fn,
    reverse_score_fn=None,
    reverse_drift_fn=None,
    T_max=1,
):
    """Compute the time-reversed diffusion.

    Args:
        forward_drift_fn: a function that takes in the state and time and returns the drift.
        forward_diffusion_scale_fn: a function that takes in the state and time and returns the diffusion.
        revesrse_score_fn: a function that takes in the state and time and returns the score.
        reverse_drift_fn: a function that takes in the state and time and returns the drift.
    """
    # exclusive or
    assert (reverse_drift_fn is not None) != (reverse_score_fn is not None)

    # rev_f = f(x, T - t)
    reverse_diffusion_scale_fn = reverse_time_argument(
        forward_diffusion_scale_fn, T_max=T_max, argnum=1
    )

    if reverse_score_fn is not None:
        reverse_forward_drift_fn = reverse_time_argument(
            forward_drift_fn, T_max=T_max, argnum=1
        )

        def reverse_drift_fn(x, t):
            return -reverse_forward_drift_fn(x, t) + ops.batch_mul(
                reverse_score_fn(x, t), reverse_diffusion_scale_fn(x, t) ** 2
            )

    return reverse_drift_fn, reverse_diffusion_scale_fn


def probability_flow_ode(drift_fn):
    """Return probability flow ODE.

    Args:
        f: a function that takes in the state and time and returns the drift.
    """

    def ode_fn(x, t):
        return 0.5 * drift_fn(x, t)

    return ode_fn


# if __name__ == "__main__":
#     import jax
#     import jax.numpy as jnp
#     from diffusion import diffusion_utils as diftils
#     from diffusion import euler_maruyama as em_sampler

#     rng = jax.random.PRNGKey(0)
#     data_shape = (16, 16, 3)
#     B = 128
#     x_init = jax.random.normal(rng, (B, *data_shape), dtype=jnp.float32)
#     t_init = 0.0

#     num_steps = 1000
#     step_schedule = jnp.ones(num_steps) / num_steps

#     f = lambda x, t: 50 - x
#     g = lambda x, t: jnp.sqrt(2) * jnp.ones(x.shape[0])

#     beta_t = lambda t: 1
#     f, g = diftils.rescale_diffusion(f, g, beta_t)

#     sampler = jax.jit(em_sampler.euler_maruyama, static_argnames=["f", "g"])

#     diffusion_state = sampler(rng, x_init, t_init, step_schedule, f, g)
#     final_samples_x = diffusion_state.x_t
#     final_samples_x = final_samples_x.reshape(-1)
#     print(jnp.mean(final_samples_x), jnp.std(final_samples_x))

#     brownian_bridge_drift = get_forward_brownian_bridge(x_T=0.0, T=1.0)
#     diffusion_state = sampler(
#         rng, x_init, t_init, step_schedule, brownian_bridge_drift, g
#     )
#     final_samples_x = diffusion_state.x_t
#     final_samples_x = final_samples_x.reshape(-1)
#     print(jnp.mean(final_samples_x), jnp.std(final_samples_x))

#     def print_arg1(x, t):
#         print(t)

#     def print_arg0(t, x):
#         print(t)

#     reverse_time(print_arg1)(1, 2)
#     reverse_time(print_arg0)(1, 2)
