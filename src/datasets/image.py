import jax
import tensorflow as tf
import tensorflow_datasets as tfds

CIFAR10 = "CIFAR10"
SVHN = "SVHN"
CELEBA = "CELEBA"
LSUN = "LSUN"

IMAGE_DATASETS = [CIFAR10, SVHN, CELEBA, LSUN]


def get_data_scaler(center_data):
    """Data normalizer. Assume data are always in [0, 1]."""
    if center_data:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(center_data):
    """Inverse data normalizer."""
    if center_data:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC,
    )
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_image_dataset(
    dataset_name,
    image_size=None,
    random_flip=None,
    data_category=None,
    uniform_dequantization=False,
):
    """Create data loaders for training and evaluation.
    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      additional_dim: An integer or `None`. If present, add one additional dimension to the output data,
        which equals the number of steps jitted together.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.
    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    shuffle_buffer_size = 10000

    # Create dataset builders for each dataset.
    if dataset_name == CIFAR10:
        dataset_builder = tfds.builder("cifar10")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [image_size, image_size], antialias=True)

    elif dataset_name == SVHN:
        dataset_builder = tfds.builder("svhn_cropped")
        train_split_name = "train"
        eval_split_name = "test"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [image_size, image_size], antialias=True)

    elif dataset_name == CELEBA:
        dataset_builder = tfds.builder("celeb_a")
        train_split_name = "train"
        eval_split_name = "validation"

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = central_crop(img, 140)
            img = resize_small(img, image_size)
            return img

    elif dataset_name == LSUN:
        dataset_builder = tfds.builder(f"lsun/{data_category}")
        train_split_name = "train"
        eval_split_name = "validation"

        if image_size == 128:

            def resize_op(img):
                img = tf.image.convert_image_dtype(img, tf.float32)
                img = resize_small(img, image_size)
                img = central_crop(img, image_size)
                return img

        else:

            def resize_op(img):
                img = crop_resize(img, image_size)
                img = tf.image.convert_image_dtype(img, tf.float32)
                return img

    # elif tfrecords_path is not None:
    #     dataset_builder = tf.data.TFRecordDataset(tfrecords_path)
    #     train_split_name = eval_split_name = "train"

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not yet supported.")

    # # Customize preprocess functions for each dataset.
    # if dataset_name in ["FFHQ", "CelebAHQ"]:

    #     def preprocess_fn(d):
    #         sample = tf.io.parse_single_example(
    #             d,
    #             features={
    #                 "shape": tf.io.FixedLenFeature([3], tf.int64),
    #                 "data": tf.io.FixedLenFeature([], tf.string),
    #             },
    #         )
    #         data = tf.io.decode_raw(sample["data"], tf.uint8)
    #         data = tf.reshape(data, sample["shape"])
    #         data = tf.transpose(data, (1, 2, 0))
    #         img = tf.image.convert_image_dtype(data, tf.float32)
    #         if random_flip and not evaluation:
    #             img = tf.image.random_flip_left_right(img)
    #         if uniform_dequantization:
    #             img = (
    #                 tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
    #             ) / 256.0
    #         return dict(image=img, label=None)

    def preprocess_fn(d):
        """Basic preprocessing function scales data to [0, 1) and randomly flips."""
        img = resize_op(d["image"])
        if random_flip:
            img = tf.image.random_flip_left_right(img)
        if uniform_dequantization:
            img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0) / 256.0

        return dict(x=img, label=d.get("label", None))

    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        # if isinstance(dataset_builder, tfds.core.DatasetBuilder):
        dataset_builder.download_and_prepare()
        ds = dataset_builder.as_dataset(
            split=split, shuffle_files=True, read_config=read_config
        )
        # else:
        #     ds = dataset_builder.with_options(dataset_options)
        ds = ds.repeat()
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    return train_ds, eval_ds
