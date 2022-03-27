import tensorflow as tf
import numpy as np


def normalize_images(images: tf.Tensor) -> tf.Tensor:
    return (tf.cast(images, tf.float32) - 127.5) / 127.5


def unnormalize_images(images: tf.Tensor) -> tf.Tensor:
    return tf.cast(images * 127.5 + 127.5, tf.uint8)


def log2(x: int) -> int:
    return int(np.log2(x))


def stage_of_resolution(res: int) -> int:
    print("This is working")
    return log2(res) - 2


def is_valid_resolution(res: int) -> bool:
    res_log2 = log2(res)
    return 2 ** res_log2 == res and res >= 4


def assert_valid_resolution(res: int) -> None:
    if not is_valid_resolution(res):
        raise RuntimeError(f"Invalid resolution: {res} (must be a power of 2 no less than 4)")


def resolution_of_stage(stage: int) -> int:
    return 2 ** (stage + 2)


def filters_for(resolution) -> int:
    res_log2 = log2(resolution)
    return num_filters(res_log2 - 1)


def num_filters(stage, fmap_base=8192, fmap_decay=1.0, fmap_max=512):
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


## He Initializer as used in the paper
class HeInitializer(tf.keras.initializers.VarianceScaling):
    def __init__(self, gain=2.0):
        super().__init__(scale=gain)

    @classmethod
    def get_constant(cls, shape: tf.TensorShape, gain: float = 2, fan_in: int = None) -> tf.Tensor:
        if fan_in is None:
            fan_in = np.prod([d.value for d in shape[:-1]])
        # He-init scaling factor
        std = np.sqrt(gain / fan_in)
        return tf.constant(std, dtype=float, name="wscale")