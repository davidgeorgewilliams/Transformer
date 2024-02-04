import tensorflow as tf

from activations import prelu
from attention import multi_head_attention
from common import batch_matmul
from normalization import layer_normalization


def encoder_layer(inputs,
                  internal_size=512,
                  num_heads=8,
                  regularizer=None,
                  dropout_rate=None,
                  mask=None):
    with tf.variable_scope("self_attention"):
        self_attention = multi_head_attention([inputs, inputs, inputs],
                                              internal_size=internal_size,
                                              mask=mask,
                                              num_heads=num_heads,
                                              output_size=inputs.shape[2].value,
                                              regularizer=regularizer)

    with tf.variable_scope("residual_0"):
        residual_0 = tf.add(inputs, self_attention)

    with tf.variable_scope("layer_norm_0"):
        layer_norm_0 = layer_normalization(residual_0, regularizer=regularizer)

    with tf.variable_scope("dense_layer_0"):
        weights = tf.get_variable(initializer=tf.glorot_uniform_initializer(),
                                  name="weights",
                                  regularizer=regularizer,
                                  shape=[layer_norm_0.shape[2], layer_norm_0.shape[2]])

        dense_layer_0 = batch_matmul(layer_norm_0, weights)

    with tf.variable_scope("activation"):
        activation = prelu(dense_layer_0, regularizer=regularizer)

    with tf.variable_scope("dense_layer_1"):
        weights = tf.get_variable(initializer=tf.glorot_uniform_initializer(),
                                  name="weights",
                                  regularizer=regularizer,
                                  shape=[activation.shape[2], activation.shape[2]])

        dense_layer_1 = batch_matmul(activation, weights)

    with tf.variable_scope("residual_1"):
        residual_1 = tf.add(layer_norm_0, dense_layer_1)

    with tf.variable_scope("layer_norm_1"):
        layer_norm_1 = layer_normalization(residual_1, regularizer=regularizer)

    if dropout_rate is not None:
        layer_norm_1 = tf.nn.dropout(layer_norm_1, rate=dropout_rate)

    return tf.identity(layer_norm_1, name="output")


def encoder_stack(inputs,
                  internal_size=512,
                  num_heads=8,
                  num_layers=6,
                  regularizer=None,
                  dropout_rate=None,
                  mask=None):
    output = inputs

    for i in range(num_layers):
        with tf.variable_scope("encoder_{}".format(i)):
            output = encoder_layer(output,
                                   dropout_rate=dropout_rate,
                                   internal_size=internal_size,
                                   mask=mask,
                                   num_heads=num_heads,
                                   regularizer=regularizer)

    return tf.identity(output, name="output")
