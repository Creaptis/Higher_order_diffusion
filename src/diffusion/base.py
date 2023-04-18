import jax
import jax.numpy as jnp
import flax

from data_types import RNG, Array
from typing import Optional, Callable, Union
from collections import OrderedDict


@flax.struct.dataclass
class Diffusion:
    """Class for keeping track of an item in inventory."""

    drift_fn: Callable = flax.struct.field(pytree_node=False)
    diffusion_scale_fn: Callable = flax.struct.field(pytree_node=False)


@flax.struct.dataclass
class DiffusionState(OrderedDict):
    x_t: Array
    mean_t: Array
    t: Array
    sigma_t: Optional[Array] = None
    rng: Optional[RNG] = None
    noise: Optional[Array] = None
    x_0: Optional[Array] = Array

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def update(self, **kwargs):
        return self.replace(**kwargs)


def init_diffusion_state(
    x_init: Array, t_init: Array = 0.0, rng: Optional[jax.random.KeyArray] = None
) -> DiffusionState:
    return DiffusionState(
        rng=rng,
        x_t=x_init,
        mean_t=x_init,
        t=t_init,
        noise=jnp.zeros_like(x_init),
        x_0=x_init,
    )


DiffusionTransitionInputs = Union[flax.struct.dataclass, jnp.ndarray]

DiffusionTransitionFn = Callable[
    [DiffusionState, DiffusionTransitionInputs], DiffusionState
]


if __name__ == "__main__":
    dif_state = DiffusionState(x_t=1, mean_t=2, t=3)
    dif_state = dif_state.replace(x_t=4)
    print(dif_state)
    # print(dif_state.get("x_t"))
    print(dif_state["x_t"])
