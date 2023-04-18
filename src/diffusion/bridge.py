import diffusion.transforms as transforms
import utils.math as ops


def get_forward_bridge(
    x_0, x_T, T, f, g, score_transition_fn, score_marginal_fn, epsilon
):
    """Compute the drift of the forward diffusion bridge.

    Args:
        x_0: the initial state.
        x_T: the terminal state.
        T: the terminal time.
        f: a function that takes in the state and time and returns the forward drift.
        g: a function that takes in the state and time and returns the forward diffusion.
        score_transition_fn: a function that takes in the state and time and returns the score of the transition density.
        score_marginal_fn: a function that takes in the state and time and returns the score of the marginal density.
        epsilon: a small positive number.
    """

    def drift_fn(x, t):
        score_marginal = score_marginal_fn(x, t)
        score_transition = score_transition_fn(x, t)

        drift = (
            f(t, x)
            + g(x, t) ** 2 * (score_marginal - score_transition)
            + epsilon * ((x_T - x) / (T - t) - (x_0 - x) / t)
        )
        return drift

    return drift_fn


def get_backward_bridge(x_0, T, forward_f, forward_g, score_fn, epsilon):
    """Compute the drift of the backward diffusion bridge.

    Args:
        x_0: the initial state.
        T: the terminal time.
        f: a function that takes in the state and time and returns the forward drift.
        g: a function that takes in the state and time and returns the forward diffusion.
        score_fn: a function that takes in the state and time and returns the score.
        epsilon: a small positive number.
    """
    reverse_f, reverse_g, score_fn = transforms.reverse_time(
        [forward_f, forward_g, score_fn], T_max=T, argnum=1
    )

    def drift_fn(x, t):
        score = score_fn(x, t)
        bridge_adjustment = epsilon * ops.batch_mul((x_0 - x), 1 / t)
        drift = -reverse_f(x, t) + score * reverse_g(x, t) ** 2 + bridge_adjustment
        return drift

    return drift_fn, reverse_g
