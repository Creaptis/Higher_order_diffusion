import jax
from typing import Tuple


def uniform_time_sampler(
    rng: jax.random.KeyArray, shape: Tuple[int], t_0: float = 0.0, T: float = 1.0
):
    return jax.random.uniform(rng, shape=shape, minval=t_0, maxval=T)
