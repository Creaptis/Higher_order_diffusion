# def loss_fn(step_rng, params, model_state, batch):

#     .....

#     return loss, model_state
import jax
import jax.numpy as jnp
import diffusion as diffusion_lib
import data_transforms as transform_lib
import models.model_utils as model_lib
import utils.math as ops


def get_prediction_target(state, model_parameteriztion="noise"):
    if model_parameteriztion == "noise":
        return -state.noise
    elif model_parameteriztion == "x_0":
        return state.x_0
    else:
        raise ValueError(f"Invalid parameterization: {model_parameteriztion}")


def get_matching_loss_fn(
    model,
    weight_fn,
    train=True,
    spatial_agg_fn=jnp.mean,
    model_parameteriztion="noise",
):
    def loss_fn(
        rng,
        params,
        model_state,
        forward_batch,
    ):
        # preamble
        # --------------------------------------------

        # get diffusion model fn, score_fn / drift_fn, v_prediction, x_0 prediction
        model_fn = model_lib.get_model_fn(model, params, model_state, train=train)

        # loss main
        # --------------------------------------------
        rng, rng_model = jax.random.split(rng)

        # x_0, v, noise, prediction
        prediction_target = get_prediction_target(
            forward_batch, model_parameteriztion=model_parameteriztion
        )
        model_prediction, model_state = model_fn(
            rng=rng_model, x=forward_batch.x_t, t=forward_batch.t
        )

        losses = jnp.square(model_prediction - prediction_target)

        # aggregate losses
        losses = jax.vmap(spatial_agg_fn)(losses)
        loss_weights = weight_fn(forward_batch)
        loss = jnp.mean(losses * loss_weights)

        return loss, model_state

    return loss_fn


def get_implicit_loss_fn(
    model,
    weight_fn,
    train=True,
    spatial_agg_fn=jnp.mean,
    hutchinson_type="gaussian",
    num_slices=1,
):
    def loss_fn(
        rng,
        params,
        model_state,
        batch,
    ):
        # get diffusion model fn, score_fn / drift_fn, v_prediction, x_0 prediction
        model_fn = model_lib.get_model_fn(model, params, model_state, train=train)

        div_fn = ops.get_div_fn(
            model_fn, hutchinson_type=hutchinson_type, argnum=0, num_slices=num_slices
        )

        # loss main
        # --------------------------------------------
        rng, rng_model, rng_div = jax.random.split(rng, 3)
        model_prediction = model_fn(rng_model, batch.x_t, batch.t)

        losses = 0.5 * jnp.square(model_prediction**2) + div_fn(
            rng_div, batch.x_t, batch.t
        )

        # aggregate losses
        losses = jax.vmap(spatial_agg_fn)(losses)
        loss_weights = weight_fn(batch)
        loss = jnp.mean(losses * loss_weights)

        return loss, model_state

    return loss_fn
