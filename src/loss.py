import numpy as np
import tensorflow as tf


def create_loss(decoder_input, decoder_labels, logits, padding_id):
    with tf.variable_scope("loss"):
        mask = tf.cast(tf.not_equal(decoder_input, padding_id), dtype=np.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=decoder_labels)
        loss = tf.reduce_sum(loss * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)
        loss = tf.reduce_mean(loss)
    return loss
