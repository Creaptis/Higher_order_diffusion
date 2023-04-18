import jax
import jax.numpy as jnp


def get_model_fn(
    model,
    params,
    model_state,
    train=True,
):
    def model_fn(x, t, cond=None, rng=None):
        model_out = model.apply(
            {"params": params, **model_state},
            x=x,
            t=t,
            train=train,
            cond=cond,
        )

        if isinstance(model_out, tuple):
            model_out, new_model_state = model_out
        else:
            new_model_state = model_state

        if train:
            return model_out, new_model_state
        else:
            return model_out

    return model_fn
