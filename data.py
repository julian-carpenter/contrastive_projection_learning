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
"""Data pipeline."""

import os
import functools
from absl import flags
from absl import logging
import socket
import data_util
import tensorflow as tf
import numpy as np

FLAGS = flags.FLAGS


def build_input_fn(ds, global_batch_size, is_training, keys, nr_examples):
    """Build input function.

    Args:
      ds: TFDS dataset.
      global_batch_size: Global batch size.
      is_training: Whether to build in training mode.
      keys: The feature keys
      nr_examples: The number of samples in the dataset

    Returns:
      A function that accepts a dict of params and returns a tuple of images and
      features, to be used as the input_fn in TPUEstimator.
    """

    def _input_fn(input_context):
        """Inner input function."""
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        logging.info('Global batch size: %d', global_batch_size)
        logging.info('Per-replica batch size: %d', batch_size)
        preprocess_fn_pretrain = get_preprocess_fn()
        preprocess_fn_finetune = get_preprocess_fn()

        def map_fn(features):
            """Produces multiple transformations of the same batch."""
            # if FLAGS.dataset == "static_helium_no_round_cart_all_meta" or "dynamic_helium_no_round_cart_all_meta":
            bunch_id = features[keys[0]]
            cart_image = features[keys[1]]
            pol_image = features[keys[2]]
            photon_energy = features[keys[3]]
            additional = features[keys[5]]
            if FLAGS.dataset == "static_helium_no_round_cart_all_meta":
                label = features[keys[4]]
            elif FLAGS.dataset == "dynamic_helium_no_round_cart_all_meta":
                label = tf.one_hot(tf.cast(features[keys[4]], tf.int32), 3)
            elif FLAGS.dataset == "maloja_cplr_all_meta_3e5_threshold":
                label = tf.one_hot(tf.cast(features[keys[4]], tf.int32), 8)
            else:
                raise KeyError("{}: Unsupported dataset. Currently only 'static_helium_no_round_cart_all_meta', "
                               "'dynamic_helium_no_round_cart_all_meta', and "
                               "'maloja_cplr_all_meta_3e5_threshold' are supported.".format(FLAGS.dataset))

            if is_training and FLAGS.train_mode == 'pretrain':
                if FLAGS.contrastive_mode == "polcart":
                    image = pol_image
                    image_t = cart_image
                elif FLAGS.contrastive_mode == "cartpol":
                    image = cart_image
                    image_t = pol_image
                elif FLAGS.contrastive_mode == "polpol":
                    image = pol_image
                    image_t = pol_image
                elif FLAGS.contrastive_mode == "cartcart":
                    image = cart_image
                    image_t = cart_image
                else:
                    raise KeyError("Valid options are: polpol, "
                                   "cartcart, polcart, and cartpol'")

                image_t = preprocess_fn_pretrain(image_t)
                image = preprocess_fn_pretrain(image)
                image = tf.concat([image, image_t], -1)
            else:
                image = preprocess_fn_finetune(pol_image)
                image_t = preprocess_fn_finetune(cart_image)

            return image, (label, image_t, bunch_id, photon_energy, additional)

        logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
        dataset = ds
        if is_training:
            options = tf.data.Options()
            options.experimental_deterministic = False
            options.experimental_slack = True
            dataset = dataset.with_options(options)
            dataset = dataset.shuffle(int(1.25 * nr_examples))
            dataset = dataset.repeat(-1)
        dataset = dataset.batch(batch_size, drop_remainder=is_training,
                                num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if FLAGS.cache_dataset:
            dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    return _input_fn


def build_distributed_dataset(batch_size, is_training, strategy, nr_examples):
    data_url = "https://share.phys.ethz.ch/~nux/datasets/{}.npz".format(FLAGS.dataset)
    if FLAGS.dataset == "static_helium_no_round_cart_all_meta":
        keys = ['bunch_id', 'cart_image', 'image', 'photon_energy', 'label', 'radius']
        file_keys = ['images', 'labels', 'cart_images', 'photon_energy', 'bunch_id', 'radius']
        md5hash = "e0a1ac6ec7b497ffc4a76e3bac2f8369"
    elif FLAGS.dataset == "dynamic_helium_no_round_cart_all_meta":
        keys = ['bunch_id', 'cart_image', 'image', 'photon_energy', 'photon_energies_categorical', "delays"]
        file_keys = ['images', 'cart_images', 'photon_energies', 'bunch_ids', 'photon_energies_categorical', "delays"]
        md5hash = "1f7a6c255c5ab0cc08a2c66486169cd5"
    elif FLAGS.dataset == "maloja_cplr_all_meta_3e5_threshold":
        keys = ['bunch_id', 'cart_image', 'image', 'photon_energy', 'label', 'radius']
        file_keys = ['images', 'cart_images', 'pid', 'xenon_concentration', 'photon_energy', 'radial_extension']
        md5hash = "a30711c936d5bdc12432308586518328"
    else:
        raise KeyError("{}: Unsupported dataset. Currently only 'static_helium_no_round_cart_all_meta', "
                       "'dynamic_helium_no_round_cart_all_meta', and "
                       "'maloja_cplr_all_meta_3e5_threshold' are supported.".format(FLAGS.dataset))

    if socket.gethostname() == "nux-noether":
        cache_dir = os.path.join("/", "scratch", "jzimmermann")
    else:
        cache_dir = "~/.keras"

    path = tf.keras.utils.get_file(fname=FLAGS.dataset + ".npz",
                                   origin=data_url,
                                   md5_hash=md5hash,
                                   cache_dir=cache_dir
                                   )
    features = {}

    with np.load(path) as d:
        if FLAGS.dataset == "static_helium_no_round_cart_all_meta":
            features.update({keys[0]: d[file_keys[4]]})
            features.update({keys[1]: d[file_keys[2]]})
            features.update({keys[2]: d[file_keys[0]]})
            features.update({keys[3]: d[file_keys[3]]})
            features.update({keys[4]: d[file_keys[1]]})
            features.update({keys[5]: d[file_keys[5]]})
        elif FLAGS.dataset == "dynamic_helium_no_round_cart_all_meta":
            features.update({keys[0]: d[file_keys[3]]})
            features.update({keys[1]: d[file_keys[1]]})
            features.update({keys[2]: d[file_keys[0]]})
            features.update({keys[3]: d[file_keys[2]]})
            features.update({keys[4]: d[file_keys[4]]})
            features.update({keys[5]: d[file_keys[5]]})
        elif FLAGS.dataset == "maloja_cplr_all_meta_3e5_threshold":
            features.update({keys[0]: d[file_keys[2]]})
            features.update({keys[1]: np.power(d[file_keys[1]], 0.25)})
            features.update({keys[2]: np.power(d[file_keys[0]], 0.25)})
            features.update({keys[3]: d[file_keys[4]]})
            xe_classes = [0., 0.1, 0.2, 0.5, 1., 2., 5., 10.]
            label = [xe_classes.index(x) for x in d[file_keys[3]]]
            features.update({keys[4]: label})
            features.update({keys[5]: d[file_keys[5]]})

    dataset = tf.data.Dataset.from_tensor_slices(features)
    input_fn = build_input_fn(dataset, batch_size, is_training, keys, nr_examples)
    return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn():
    """Get function that accepts an image and returns a preprocessed image."""
    return functools.partial(
        data_util.preprocess_image,
        height=FLAGS.image_size,
        width=FLAGS.image_size)
