import numpy as np
import tensorflow as tf


def batch_normalization(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training)


def layer_normalization(inputs, epsilon=1e-6, regularizer=None):
    """ Layer Normalization: https://arxiv.org/abs/1607.06450 """
    with tf.variable_scope("layer_normalization"):
        gamma = tf.get_variable(initializer=tf.constant_initializer(1.0, dtype=np.float32),
                                name="gamma",
                                regularizer=regularizer,
                                shape=inputs.shape[-1:])

        beta = tf.get_variable(initializer=tf.constant_initializer(0.1, dtype=np.float32),
                               name="beta",
                               regularizer=regularizer,
                               shape=inputs.shape[-1:])

        input_mean = tf.reduce_mean(inputs, axis=-1, keepdims=True, name="mean")
        input_stdev = tf.math.reduce_std(inputs, axis=-1, keepdims=True, name="stdev")

        normalized_input = (inputs - input_mean) / (input_stdev + epsilon)

        return gamma * normalized_input + beta
