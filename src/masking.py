import numpy as np
import tensorflow as tf


def auto_regressive_mask(inputs):
    with tf.variable_scope("auto_regressive_mask"):
        sequence_length = inputs.shape[1].value
        rows = tf.expand_dims(tf.range(sequence_length), axis=1)
        columns = tf.range(sequence_length)
        mask = tf.expand_dims(tf.expand_dims(rows >= columns, axis=0), axis=0)
        return -1e10 * (1.0 - tf.cast(mask, dtype=np.float32))


def padding_mask(inputs, padding_value=0):
    with tf.variable_scope("padding_mask"):
        padding = tf.cast(tf.equal(inputs, padding_value), np.float32)
        return tf.expand_dims(tf.expand_dims(padding * -1e10, axis=1), axis=1)
