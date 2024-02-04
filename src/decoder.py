import tensorflow as tf

from activations import prelu
from attention import multi_head_attention
from common import batch_matmul
from normalization import layer_normalization


def decoder_layer(encoder_output,
                  decoder_input,
                  internal_size=512,
                  num_heads=8,
                  regularizer=None,
                  dropout_rate=None,
                  encoder_mask=None,
                  decoder_mask=None):
    with tf.variable_scope("decoder_self_attention"):
        decoder_self_attention = multi_head_attention([decoder_input, decoder_input, decoder_input],
                                                      internal_size=internal_size,
                                                      output_size=decoder_input.shape[2].value,
                                                      num_heads=num_heads,
                                                      regularizer=regularizer,
                                                      mask=decoder_mask)

    with tf.variable_scope("residual_0"):
        residual_0 = tf.add(decoder_input, decoder_self_attention)

    with tf.variable_scope("layer_norm_0"):
        layer_norm_0 = layer_normalization(residual_0, regularizer=regularizer)

    with tf.variable_scope("encoder_decoder_attention"):
        encoder_decoder_attention = multi_head_attention([layer_norm_0, encoder_output, encoder_output],
                                                         internal_size=internal_size,
                                                         output_size=layer_norm_0.shape[2].value,
                                                         num_heads=num_heads,
                                                         regularizer=regularizer,
                                                         mask=encoder_mask)

    with tf.variable_scope("residual_1"):
        residual_1 = tf.add(layer_norm_0, encoder_decoder_attention)

    with tf.variable_scope("layer_norm_1"):
        layer_norm_1 = layer_normalization(residual_1, regularizer=regularizer)

    with tf.variable_scope("dense_layer_0"):
        weights = tf.get_variable(name="weights",
                                  shape=[layer_norm_1.shape[2], layer_norm_1.shape[2]],
                                  initializer=tf.glorot_uniform_initializer(),
                                  regularizer=regularizer)

        dense_layer_0 = batch_matmul(layer_norm_1, weights)

    with tf.variable_scope("activation"):
        activation = prelu(dense_layer_0, regularizer=regularizer)

    with tf.variable_scope("dense_layer_1"):
        weights = tf.get_variable(name="weights",
                                  shape=[activation.shape[2], activation.shape[2]],
                                  initializer=tf.glorot_uniform_initializer(),
                                  regularizer=regularizer)

        dense_layer_1 = batch_matmul(activation, weights)

    with tf.variable_scope("residual_2"):
        residual_2 = tf.add(layer_norm_1, dense_layer_1)

    with tf.variable_scope("layer_norm_2"):
        layer_norm_2 = layer_normalization(residual_2, regularizer=regularizer)

    if dropout_rate is not None:
        layer_norm_2 = tf.nn.dropout(layer_norm_2, rate=dropout_rate)

    return tf.identity(layer_norm_2, name="output")


def decoder_stack(encoder_output,
                  decoder_input,
                  internal_size=512,
                  num_heads=8,
                  num_layers=6,
                  regularizer=None,
                  dropout_rate=None,
                  encoder_mask=None,
                  decoder_mask=None):
    decoder_output = decoder_input

    for i in range(num_layers):
        with tf.variable_scope("decoder_{}".format(i)):
            decoder_output = decoder_layer(encoder_output,
                                           decoder_output,
                                           internal_size=internal_size,
                                           num_heads=num_heads,
                                           regularizer=regularizer,
                                           dropout_rate=dropout_rate,
                                           encoder_mask=encoder_mask,
                                           decoder_mask=decoder_mask)

    return tf.identity(decoder_output, name="decoder_stack")
