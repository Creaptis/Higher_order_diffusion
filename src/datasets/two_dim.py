import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn import datasets

EIGHTGAUSSIANS = "8gaussians"
SCURVE = "scurve"
SWISS = "swiss"
MOON = "moon"
CIRCLE = "circle"
CHECKER = "checker"
PINWHEEL = "pinwheel"


TWO_DIM_DATASETS = [
    EIGHTGAUSSIANS,
    SCURVE,
    SWISS,
    MOON,
    CIRCLE,
    CHECKER,
    PINWHEEL,
]


def scale_x(x):
    return 2 * ((x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0)) - 0.5)


def get_two_dim_dataset(dataset_name, num_two_dim_samples=10_000):
    train_ds = _get_two_dim_dataset(
        dataset_name, num_two_dim_samples=num_two_dim_samples, split="train"
    )
    eval_ds = _get_two_dim_dataset(
        dataset_name, num_two_dim_samples=num_two_dim_samples, split="train"
    )
    return train_ds, eval_ds


def _get_two_dim_dataset(
    dataset_name, num_two_dim_samples=10_000, split="train", scale=True
):
    random_state = 0 if split == "train" else 1
    np.random.seed(random_state)
    shuffle_buffer_size = 10000
    if dataset_name == SCURVE:
        X, y = datasets.make_s_curve(
            n_samples=num_two_dim_samples, noise=0.0, random_state=random_state
        )
        init_sample = X[:, [0, 2]]
        scaling_factor = 7
        init_sample = (
            (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor
        )

    if dataset_name == SWISS:
        X, y = datasets.make_swiss_roll(
            n_samples=num_two_dim_samples, noise=0.0, random_state=random_state
        )
        init_sample = X[:, [0, 2]]
        scaling_factor = 7
        init_sample = (
            (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor
        )

    if dataset_name == MOON:
        X, y = datasets.make_moons(
            n_samples=num_two_dim_samples, noise=0.0, random_state=random_state
        )
        scaling_factor = 7.0
        init_sample = X
        init_sample = (
            (init_sample - init_sample.mean()) / init_sample.std() * scaling_factor
        )

    if dataset_name == CIRCLE:
        X, y = datasets.make_circles(
            n_samples=num_two_dim_samples,
            noise=0.0,
            random_state=random_state,
            factor=0.5,
        )
        init_sample = X * 10

    if dataset_name == CHECKER:
        x1 = np.random.rand(num_two_dim_samples) * 4 - 2
        x2_ = (
            np.random.rand(num_two_dim_samples)
            - np.random.randint(0, 2, num_two_dim_samples) * 2
        )
        x2 = x2_ + (np.floor(x1) % 2)
        x = np.concatenate([x1[:, None], x2[:, None]], 1) * 7.5
        init_sample = x

    if dataset_name == PINWHEEL:
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = num_two_dim_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes * num_per_class, 2) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        )
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        x = 7.5 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))
        init_sample = x

    if dataset_name == EIGHTGAUSSIANS:
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(num_two_dim_samples):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset *= 3
        init_sample = dataset

    init_sample = init_sample.astype(np.float32)
    if scale:
        init_sample = scale_x(init_sample)
    ds = tf.data.Dataset.from_tensor_slices(init_sample)
    ds = ds.repeat()
    ds = ds.shuffle(shuffle_buffer_size)

    def preprocess_fn(d):
        """Basic preprocessing function scales data to [0, 1) and randomly flips."""
        return dict(x=d, label=None)

    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds
