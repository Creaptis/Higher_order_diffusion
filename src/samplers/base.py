from typing import List
from data_types import Array
import diffusion.base as diffusion_lib
import jax
import jax.numpy as jnp
import flax
from utils import math as ops


def simulate_diffusion(
    diffusion_state: diffusion_lib.DiffusionState,
    transition_inputs: diffusion_lib.DiffusionTransitionInputs,
    step_fn: diffusion_lib.DiffusionTransitionFn,
    return_trajectory: bool = False,
    store_state_keys: List[str] = ["x_t", "mean_t", "t", "noise"],
):
    """Simulate multiple steps of a step function.

    Args:
        rng: a JAX PRNGKey.
        init_state: the initial state.
        transition_inputs: time indexed inputs
        step_fn: a function that takes in the state and inputs and returns the next state.
        return_trajectory: whether to store the trajectory.
    """

    def scan_fn(diffusion_state, transition_input):
        next_diffusion_state = step_fn(diffusion_state, transition_input)
        store_state = (
            {key: getattr(next_diffusion_state, key) for key in store_state_keys}
            if return_trajectory
            else ()
        )

        return next_diffusion_state, store_state

    final_state, state_trajectory = jax.lax.scan(
        f=scan_fn,
        init=diffusion_state,
        xs=transition_inputs,
        length=None,
        reverse=False,
        unroll=1,
    )

    return state_trajectory if return_trajectory else final_state


# update stored states
def _store_state(
    i,
    stored_states,
    diffusion_state,
    update_time_index,
    store_state_time_index,
    store_state_keys,
):
    update_sample_index = i == ops.batch_select(
        store_state_time_index, update_time_index, in_axes=(0, 0)
    )

    N = diffusion_state.x_t.shape[0]
    new_stored_states = {}
    for key in store_state_keys:
        updates = jax.vmap(jnp.where, in_axes=(0, 0, 0))(
            update_sample_index,
            getattr(diffusion_state, key),
            ops.batch_select(stored_states[key], update_time_index, in_axes=(1, 0)),
        )
        new_stored_states[key] = (
            stored_states[key]
            .at[update_time_index, jnp.arange(N, dtype=int)]
            .set(updates)
        )

    new_update_time_index = jnp.where(
        update_sample_index, update_time_index + 1, update_time_index
    )

    return new_stored_states, new_update_time_index


def partial_diffusion_trajectory(
    diffusion_state: diffusion_lib.DiffusionState,
    transition_inputs: diffusion_lib.DiffusionTransitionInputs,
    step_fn: diffusion_lib.DiffusionTransitionFn,
    store_state_time_index=Array,  # [Array, "N T"]
    store_state_keys: List[str] = ["x_t", "x_t_mean", "t", "noise"],
):
    """Simulate multiple steps of a step function, return subset of trajectory.

    Args:
        rng: a JAX PRNGKey.
        init_state: the initial state.
        transition_inputs: time indexed inputs
        step_fn: a function that takes in the state and inputs and returns the next state.
        store_state_time_index: time index to store states
        store_state_keys: keys to store
    """
    N, T = store_state_time_index.shape[0], store_state_time_index.shape[1]

    # sort time index of which times to store
    store_state_time_index = jnp.sort(store_state_time_index, axis=1)

    # init object to store attributes of interest, each of shape [T N *state_dim]
    stored_states = {
        key: jnp.ones((T,) + getattr(diffusion_state, key).shape)
        for key in store_state_keys
    }

    # running time index to update, per particle, start with 0 i.e. first time index
    update_time_index = jnp.zeros((N,), dtype=int)  # [Array, "N"]

    # if 0 is in time index, store initial state
    stored_states, update_time_index = _store_state(
        0,
        stored_states,
        diffusion_state,
        update_time_index,
        store_state_time_index,
        store_state_keys,
    )
    init_carry_state = {
        "i": 0,
        "diffusion_state": diffusion_state,
        "update_time_index": update_time_index,
        "stored_states": stored_states,
    }

    def scan_fn(carry_state, transition_input):
        next_diffusion_state = step_fn(carry_state["diffusion_state"], transition_input)
        i = carry_state["i"] + 1
        stored_states, update_time_index = _store_state(
            i,
            carry_state["stored_states"],
            next_diffusion_state,
            carry_state["update_time_index"],
            store_state_time_index,
            store_state_keys,
        )

        next_carry_state = {
            "i": i,
            "diffusion_state": next_diffusion_state,
            "update_time_index": update_time_index,
            "stored_states": stored_states,
        }
        return next_carry_state, ()

    final_state, _ = jax.lax.scan(
        f=scan_fn,
        init=init_carry_state,
        xs=transition_inputs,
        length=None,
        reverse=False,
        unroll=1,
    )
    stored_states = final_state["stored_states"]  # [Array, T N *state_dim]

    return stored_states


def get_noising_fn(time_sampler, state_noising_process):
    def noising_fn(init_state):
        rng_x, rng_t, new_rng = jax.random.split(init_state.rng, 3)
        init_state = init_state.replace(rng=rng_x)
        batch_size = init_state.x_0.shape[0]
        t = time_sampler(rng=rng_t, shape=(batch_size,))
        state_t = state_noising_process(diffusion_state=init_state, t=t)
        state_t = state_t.replace(rng=new_rng)
        return state_t

    return noising_fn
