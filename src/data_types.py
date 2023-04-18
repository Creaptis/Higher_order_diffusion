import jax
import jax.numpy as jnp
import jaxtyping as jtyp
import typing

Array = jtyp.Array
RNG = jax.random.PRNGKeyArray
DriftFn = typing.Callable[[jtyp.Array, jtyp.Array], jtyp.Array]
DiffusionFn = typing.Callable[[jtyp.Array, jtyp.Array], jtyp.Array]
