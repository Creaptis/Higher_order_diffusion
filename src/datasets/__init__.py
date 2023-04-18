import jax
import tensorflow as tf

from .image import IMAGE_DATASETS, get_image_dataset, CIFAR10, SVHN, CELEBA, LSUN
from .two_dim import (
    TWO_DIM_DATASETS,
    get_two_dim_dataset,
    SCURVE,
    MOON,
    CIRCLE,
    SWISS,
    PINWHEEL,
    EIGHTGAUSSIANS,
    CHECKER,
)


def batch_dataset(
    ds,
    batch_size,
    additional_dim=None,
):
    # Compute batch size for this worker.
    if batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch sizes ({batch_size} must be divided by"
            f"the number of devices ({jax.device_count()})"
        )

    per_device_batch_size = batch_size // jax.device_count()
    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000

    # Create additional data dimension when jitting multiple steps together
    if additional_dim is None:
        batch_dims = [jax.local_device_count(), per_device_batch_size]
    else:
        batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]

    prefetch_size = tf.data.experimental.AUTOTUNE
    for batch_size in reversed(batch_dims):
        ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)


def get_dataset(
    dataset_name,
    train_batch_size,
    eval_batch_size,
    num_jit_steps=1,
    image_size=None,
    random_flip=None,
    data_category=None,
    uniform_dequantization=False,
    num_two_dim_samples=10_000,
):
    """Return training and evaluation/test datasets from config files."""
    if dataset_name in IMAGE_DATASETS:
        train_ds, eval_ds = get_image_dataset(
            dataset_name,
            image_size=image_size,
            random_flip=random_flip,
            data_category=data_category,
            uniform_dequantization=uniform_dequantization,
        )
    elif dataset_name in TWO_DIM_DATASETS:
        train_ds, eval_ds = get_two_dim_dataset(dataset_name, num_two_dim_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_ds = batch_dataset(train_ds, train_batch_size, additional_dim=num_jit_steps)
    eval_ds = batch_dataset(train_ds, eval_batch_size, additional_dim=None)
    return train_ds, eval_ds
