from typing import Any, Tuple
import flax
import optax
import jax
import jax.numpy as jnp
import utils.config as config_lib


@flax.struct.dataclass
class TrainState:
    step: int
    params: Any
    model_state: Any
    opt_state: Any
    ema_params: Any


def ema_step(ema_params, new_params, ema_rate):
    new_params_ema = jax.tree_map(
        lambda p_ema, p: p_ema * ema_rate + p * (1.0 - ema_rate),
        ema_params,
        new_params,
    )
    return new_params_ema


def get_train_step_fn(
    loss_fn,
    optimizer,
    ema_rate,
    noising_fn=None,
):
    """Create a one-step training/evaluation function.
    Args:
      loss_fn: loss function to compute
      optimizer: optax optimizer to use
      ema_rate: exponential moving average rate
      train: whether to train or evaluate

    Returns:
      A one-step function for training or evaluation.
    """

    def step_fn(carry_state: Tuple[jax.random.KeyArray, TrainState], batch: dict):
        """Running one step of training or evaluation.
        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.
        Args:
          carry_state: A tuple (JAX random state, NamedTuple containing the training state).
          batch: A mini-batch of training/evaluation data.
        Returns:
          new_carry_state: The updated tuple of `carry_state` and rng.
          loss: The average loss value of this state.
        """

        (rng, train_state) = carry_state
        rng, step_rng = jax.random.split(rng)
        grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)

        if noising_fn is not None:
            batch = noising_fn(batch)

        # apply loss function and compute grads
        (loss, new_model_state), grad = grad_fn(
            step_rng, train_state.params, train_state.model_state, batch
        )
        grad = jax.lax.pmean(grad, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")

        # apply optimizer
        updates, new_opt_state = optimizer.update(grad, train_state.opt_state)
        new_params = optax.apply_updates(train_state.params, updates)

        # exponential moving average
        new_params_ema = ema_step(train_state.ema_params, new_params, ema_rate)

        # update train state
        step = train_state.step + 1
        new_train_state = train_state.replace(
            step=step,
            opt_state=new_opt_state,
            model_state=new_model_state,
            params=new_params,
            ema_params=new_params_ema,
        )

        # record loss and update carry state
        store_state = (loss, step)
        new_carry_state = (rng, new_train_state)

        return new_carry_state, store_state

    return step_fn


if __name__ == "__main__":
    # PYTHONPATH=/Users/jamesthornton/DiffusionBridge/db_jax/src
    optimizer_config = config_lib.Config(learning_rate=2e-4, b1=0.9, b2=0.999, eps=1e-8)
    scheduler_config = config_lib.Config(init_value=1.0, decay_steps=10_000, alpha=0.0)

    schedule_fn = optax.cosine_decay_schedule(**scheduler_config.as_dict())
    optimizer = optax.chain(
        optax.adam(**optimizer_config.as_dict()), optax.scale_by_schedule(schedule_fn)
    )

    from training.losses import get_matching_loss_fn

    loss_fn = get_matching_loss_fn("l2", 0.1)
    step_fn = get_train_step_fn(loss_fn, optimizer, 0.99)
