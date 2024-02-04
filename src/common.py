import tensorflow as tf


def batch_matmul(inputs, kernel):
    with tf.name_scope("batch_matmul"):
        reshaped = tf.reshape(inputs, [-1, kernel.shape[0]])
        multiplied = tf.matmul(reshaped, kernel)
        return tf.reshape(multiplied, [-1, inputs.shape[1], kernel.shape[1]])


def get_regularization_loss():
    return tf.losses.get_regularization_loss()
