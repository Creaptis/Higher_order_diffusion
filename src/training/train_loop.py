import diffusion as diffusion_lib
import diffusion.beta_schedule as schedule_lib
import samplers as sampler_lib
import training.time_sampler as time_sampler_lib
import datasets as dataset_lib
import training.losses as loss_lib
import training.train as train_lib
import utils.config as config_lib
import models.mlp as model_lib
import utils.logging as loglib

import flax
import jax
import jax.numpy as jnp
import optax
import functools
import copy


# input parameters
# --------------------------------------------
# training
batch_size = 128
seed = 0
rng = jax.random.PRNGKey(seed)
num_training_iterations = 10_000
ema_rate = 0.99
num_devices = jax.local_device_count()

# optimizer
optimizer_config = config_lib.Config(learning_rate=2e-4, b1=0.9, b2=0.999, eps=1e-8)
scheduler_config = config_lib.Config(init_value=1.0, decay_steps=10_000, alpha=0.0)

# dataset
# dataset_name = "CIFAR10"
# image_size = 32
# data_shape = (image_size, image_size, 3)
dataset_name = dataset_lib.SCURVE
data_shape = (2,)

# diffusion
T_max = 1.0
t_0 = 1e-4
num_timesteps = 1000

beta_max = 1.0
beta_min = 1e-4

mean_T = jnp.zeros(data_shape)
sigma_T = jnp.ones(data_shape)


# Set up
# --------------------------------------------
# --------------------------------------------

# data
# --------------------------------------------
train_ds, eval_ds = dataset_lib.get_dataset(
    dataset_name,
    train_batch_size=batch_size,
    eval_batch_size=batch_size,
    num_jit_steps=None,
    image_size=None,
    random_flip=None,
    data_category=None,
    uniform_dequantization=False,
    num_two_dim_samples=10_000,
)
train_iter = iter(train_ds)

# set up data transform
transform_fn = lambda x: x

# forward diffusion
# --------------------------------------------
beta_schedule = schedule_lib.LinearSchedule(
    beta_0=beta_min, beta_T=beta_max, t_0=t_0, T=T_max
)
state_sampler = sampler_lib.get_ornstein_uhlenbeck_sampler(
    mean_T=mean_T, sigma_T=sigma_T, beta_schedule=beta_schedule
)
drift_fn, diffusion_scale_fn = diffusion_lib.get_ornstein_uhlenbeck_diffusion(
    mean_T=mean_T, sigma_T=sigma_T, beta_schedule=beta_schedule
)

# set up time sampler
time_sampler = functools.partial(
    time_sampler_lib.uniform_time_sampler, t_0=t_0, T=T_max
)
noising_fn = sampler_lib.get_noising_fn(time_sampler, state_sampler)

# model
# --------------------------------------------
model = model_lib.MLPDiffusionModel()

# training
# --------------------------------------------
# set up optimizer
schedule_fn = optax.cosine_decay_schedule(**scheduler_config.as_dict())
optimizer = optax.chain(
    optax.adam(**optimizer_config.as_dict()), optax.scale_by_schedule(schedule_fn)
)

# setup training step
loss_fn = loss_lib.get_matching_loss_fn(
    model=model,
    weight_fn=lambda x: 1.0,
)
train_step = train_lib.get_train_step_fn(
    loss_fn=loss_fn,
    optimizer=optimizer,
    ema_rate=ema_rate,
    noising_fn=noising_fn,
)

# init training state
# --------------------------------------------
dummy_x = jnp.ones((1, *data_shape))
dummy_t = jnp.ones((1,))
rng, init_rng = jax.random.split(rng)

vars = model.init(rng, dummy_x, dummy_t)
model_state, params = vars.pop("params")
opt_state = optimizer.init(params)


# setup train state
train_state = train_lib.TrainState(
    step=0,
    params=params,
    model_state=model_state,
    opt_state=opt_state,
    ema_params=copy.deepcopy(params),
)

# parallel across devices
# p_train_step = jax.pmap(
#     functools.partial(jax.lax.scan, train_step), axis_name="batch", donate_argnums=1
# )
p_train_step = jax.pmap(train_step, axis_name="batch", donate_argnums=1)
pstate = flax.jax_utils.replicate(train_state)

# sampling
# --------------------------------------------

# checkpoints
# --------------------------------------------

# training loop
# --------------------------------------------

for it in range(num_training_iterations):
    rng, train_rng, time_rng, init_rng = jax.random.split(rng, 4)
    batch = jax.tree_map(lambda x: x._numpy(), next(train_iter))
    x = batch["x"]
    z = transform_fn(x)

    init_t = jnp.zeros((num_devices, batch_size))
    init_rng = jax.random.split(init_rng, num_devices)
    init_state = diffusion_lib.init_diffusion_state(
        x_init=z, t_init=init_t, rng=init_rng
    )

    # rng per device
    train_rng = jnp.asarray(jax.random.split(train_rng, num_devices))

    # train step
    carry_state = (train_rng, pstate)
    (_, pstate), store_state = p_train_step(
        carry_state,
        init_state,
    )

    # metrics
    train_state = flax.jax_utils.unreplicate(pstate)
    if train_state.step % 10 == 0:
        (loss, step) = store_state
        print_metrics = {"step": step, "loss": loss}
        loglib.print_metrics(print_metrics)
