import numpy as np
import tensorflow as tf


def prelu(inputs, initial_alpha=0.01, regularizer=None):
    """ PReLU Activation: https://arxiv.org/abs/1502.01852 """
    with tf.variable_scope("PRelU", reuse=tf.AUTO_REUSE):
        alpha = tf.get_variable(initializer=tf.constant_initializer(initial_alpha, dtype=np.float32),
                                name="alpha",
                                regularizer=regularizer,
                                shape=inputs.shape[1:])
        return tf.nn.relu(inputs) - alpha * tf.nn.relu(-1.0 * inputs)
