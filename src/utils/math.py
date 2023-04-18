import jax
from typing import Tuple, Sequence, Callable
from jaxtyping import Array, Float
import jax.numpy as jnp


def batch_mul(
    x: Float[Array, "B ..."], y: Float[Array, "B ..."], in_axes: Tuple[int] = (0, 0)
) -> Float[Array, "B ..."]:
    return jax.vmap(lambda x, y: x * y, in_axes=in_axes)(x, y)


def batch_add(
    x: Float[Array, "B ..."], y: Float[Array, "B ..."], in_axes: Tuple[int] = (0, 0)
) -> Float[Array, "B ..."]:
    return jax.vmap(lambda x, y: x + y, in_axes=in_axes)(x, y)


def batch_select(x: Array, i: Array, in_axes) -> Array:
    select = jax.vmap(lambda x, i: x[i], in_axes=in_axes)
    return select(x, i)


def get_div_fn(
    func, hutchinson_type: str = "None", argnum: int = 0, num_slices: int = 1
):
    """Euclidean divergence of the function."""
    if hutchinson_type == "None":
        return get_exact_div_fn(func, argnum)
    else:
        return get_hutchinson_div_fn(func, argnum, hutchinson_type, num_slices)


def get_hutchinson_div_fn(fn, argnum, hutchinson_type, num_slices=1):
    def div_fn(rng, *args, **kwargs):
        rngs = jnp.array(jax.random.split(rng, num_slices))
        divs = jax.vmap(
            get_single_hutchinson_div_fn(fn, argnum, hutchinson_type),
            in_axes=(0, None, None),
        )(rngs, *args, **kwargs)
        return jnp.mean(divs, axis=0)

    return div_fn


def get_single_hutchinson_div_fn(fn, argnum=0, hutchinson_type="gaussian"):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(rng, *args, **kwargs):
        shape = list(args)[argnum].shape

        if hutchinson_type.lower() == "gaussian":
            eps = jax.random.normal(rng, shape)
        elif hutchinson_type.lower() == "rademacher":
            eps = jax.random.randint(rng, shape, minval=0, maxval=2).astype(jnp.float32)
            eps = eps * 2 - 1
        else:
            raise ValueError("Unknown Hutchinson type: {}".format(hutchinson_type))
        # spherical uniform to do

        grad_fn = lambda *args, **kwargs: jnp.sum(fn(*args, **kwargs * eps))
        grad_fn_eps = jax.grad(grad_fn, argnums=argnum)(*args, **kwargs)
        return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(shape))))

    return div_fn


def get_exact_div_fn(fn, argnum: int = 0):
    @jax.vmap
    def div_fn(rng, *args, **kwargs):
        del rng  # unused
        jac = jax.jacrev(fn, argnums=argnum)(*args, **kwargs)
        return jnp.trace(jac)

    return div_fn
