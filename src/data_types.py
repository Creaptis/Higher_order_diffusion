import jax
import jax.numpy as jnp
import typing

Array = jnp.ndarray
RNG = jax.random.PRNGKeyArray
DriftFn = typing.Callable[[Array, Array], Array]
DiffusionFn = typing.Callable[[Array, Array], Array]
