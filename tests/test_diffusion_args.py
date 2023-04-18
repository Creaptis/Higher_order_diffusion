import jax


def wrapper(func):
    def f(*args, **kwargs):
        if "diffusion" in kwargs:
            kwargs["f"] = kwargs["diffusion"]
            kwargs.pop("diffusion")
        return func(*args, **kwargs)

    return f


@wrapper
def test(f, g, **kwargs):
    return f + g


jit_test = jax.jit(test)
(jit_test(f=1, g=1, diffusion=6), jit_test(f=1, g=1), jit_test(g=1, diffusion=6))
