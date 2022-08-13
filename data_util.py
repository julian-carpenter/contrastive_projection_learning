# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Data preprocessing and augmentation."""

from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS


def preprocess_image(image, height, width):
    """Preprocesses the given image.

    Args:
      image: `Tensor` representing an image of arbitrary size.
      height: Height of output image.
      width: Width of output image.

    Returns:
      A converted and resized image `Tensor` of range [0, 1].
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return tf.image.resize(image, [height, width],
                           method=tf.image.ResizeMethod.BICUBIC)


@tf.function()
def fill_boxes(inpimg, val=0.):
    inp, img = inpimg[0], inpimg[1]
    img = tf.cast(img, tf.float32)

    a, b, c, d = inp[0], inp[1], inp[2], inp[3]

    aa = tf.range(tf.squeeze(a), tf.squeeze(c))
    bb = tf.range(tf.squeeze(b), tf.squeeze(d))

    aa, bb = aa[None, :, None], bb[:, None, None]
    ind = tf.concat([aa + tf.zeros_like(bb),
                     tf.zeros_like(aa) + bb], axis=2)

    ind = tf.reshape(ind, [-1, 2])
    ind_shape = tf.shape(ind)
    update_shape = (ind_shape[0], 1)
    upd = tf.multiply(tf.ones(update_shape, dtype=img.dtype), val)

    x = tf.tensor_scatter_nd_update(img, tf.cast(ind, tf.int64), upd)
    return x


class RandomResizedCrop(tf.keras.layers.Layer):
    def __init__(self, scale, ratio, **kwargs):
        super().__init__(**kwargs)
        # area-range of the cropped part: (min area, max area), uniform sampling
        self.scale = scale
        # aspect-ratio-range of the cropped part: (log min ratio, log max ratio), log-uniform sampling
        if ratio[0] == ratio[1] == 1:
            self.adjust_ratio = False
        else:
            self.adjust_ratio = True

        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]

            # independently sampled scales and ratios for every image in the batch
            random_scales = tf.random.uniform(
                (batch_size,), self.scale[0], self.scale[1]
            )
            if self.adjust_ratio:
                random_ratios = tf.exp(
                    tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
                )
                # corresponding height and widths, clipped to fit in the image
                new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
                new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)
            else:
                new_widths = tf.clip_by_value(tf.sqrt(random_scales), 0, 1)
                new_heights = tf.identity(new_widths)

            # random anchors for the crop bounding boxes
            height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
            width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

            # assemble bounding boxes and crop
            bounding_boxes = tf.stack(
                [
                    height_offsets,
                    width_offsets,
                    height_offsets + new_heights,
                    width_offsets + new_widths,
                ],
                axis=1,
            )
            images = tf.image.crop_and_resize(
                images, bounding_boxes, tf.range(batch_size), (height, width)
            )

        return images


class RandomFill(tf.keras.layers.Layer):
    def __init__(self, value, scale, ratio, **kwargs):
        super().__init__(**kwargs)
        # area-range of the cropped part: (min area, max area), uniform sampling
        self.scale = scale
        # aspect-ratio-range of the cropped part: (log min ratio, log max ratio), log-uniform sampling
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

        self.fn = lambda a: fill_boxes(a, val=value)

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]

            # independently sampled scales and ratios for every image in the batch
            random_scales = tf.random.uniform(
                (batch_size,), self.scale[0], self.scale[1]
            )
            random_ratios = tf.exp(
                tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
            )

            # corresponding height and widths, clipped to fit in the image
            new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
            new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)

            # random anchors for the crop bounding boxes
            height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
            width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

            # assemble bounding boxes and crop
            bounding_boxes = tf.stack(
                [
                    tf.math.round(tf.math.multiply(height_offsets, tf.cast(height, tf.float32))),
                    tf.math.round(tf.math.multiply(width_offsets, tf.cast(width, tf.float32))),
                    tf.math.round(tf.math.multiply(height_offsets + new_heights, tf.cast(height, tf.float32))),
                    tf.math.round(tf.math.multiply(width_offsets + new_widths, tf.cast(width, tf.float32))),
                ],
                axis=1,
            )

            images = tf.map_fn(self.fn, [bounding_boxes, images],
                               fn_output_signature=tf.float32)

        return images


class RandomJitterAffine(tf.keras.layers.Layer):
    def __init__(self, brightness=0., jitter=0., **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.jitter, maxval=self.jitter
            )

            jitter_transforms = (
                    tf.eye(1, batch_shape=[batch_size, 1]) * brightness_scales
                    + jitter_matrices
            )
            images = tf.clip_by_value(tf.matmul(images, tf.cast(jitter_transforms, images.dtype)), 0, 1)
        return images


# Image augmentation module
def gpu_augmenter(in_shape, brightness, jitter, scale, ratio,
                  fill_scale, fill_ratio, fill_value, strength):
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=in_shape),
            tf.keras.layers.RandomFlip("horizontal", name="RandomFlip"),
            RandomResizedCrop(scale=scale, ratio=ratio, name="RandomResizedCrop"),
            RandomJitterAffine(brightness, jitter, name="RandomJitterAffine"),
            tf.keras.layers.RandomRotation(strength, fill_mode="constant",
                                           fill_value=fill_value,
                                           name="RandomRotation"),
            RandomFill(value=fill_value, scale=fill_scale,
                       ratio=fill_ratio, name="RandomFill"),
            tf.keras.layers.RandomTranslation(strength / 2, strength / 2, fill_mode="constant",
                                              fill_value=fill_value,
                                              name="RandomTranslation"),
        ]
    )
