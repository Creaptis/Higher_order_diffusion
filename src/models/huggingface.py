from diffusers import FlaxUNet2DConditionModel
import jax
import jax.numpy as jnp

if __name__ == "__main__":
    model_config = dict(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        only_cross_attention=False,
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        attention_head_dim=8,
        cross_attention_dim=1280,
        dropout=0.0,
        use_linear_projection=False,
        dtype=jnp.float32,
        flip_sin_to_cos=True,
        freq_shift=0,
    )

    model = FlaxUNet2DConditionModel(**model_config)

    batch_size = 32
    channel = 3
    height = 32
    width = 32

    sample_shape = (1, model.in_channels, model.sample_size, model.sample_size)
    sample = jnp.zeros(sample_shape, dtype=jnp.float32)
    timesteps = jnp.ones((1,), dtype=jnp.int32)
    encoder_hidden_states = jnp.zeros(
        (1, 1, model.cross_attention_dim), dtype=jnp.float32
    )
    rng = jax.random.PRNGKey(0)
    params = model.init_weights(rng)

    model_pred = model.apply(
        {"params": params}, sample, timesteps, encoder_hidden_states, train=True
    ).sample
    print(jax.tree_map(lambda x: x.shape, model_pred))
