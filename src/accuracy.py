import numpy as np
import tensorflow as tf


def create_accuracy(decoder_labels, padding_id, softmax):
    with tf.variable_scope("accuracy"):
        mask = tf.cast(tf.not_equal(decoder_labels, padding_id), dtype=np.float32)
        argmax = tf.argmax(softmax, axis=-1, output_type=np.int32)
        accuracy = tf.cast(tf.equal(decoder_labels, argmax), dtype=np.float32)
        accuracy = tf.reduce_sum(accuracy * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)
        accuracy = tf.reduce_mean(accuracy)
    return accuracy
