import numpy as np
import tensorflow as tf

from common import batch_matmul
from decoder import decoder_stack
from embedding import positional_embedding
from encoder import encoder_stack
from masking import auto_regressive_mask, padding_mask


def transformer(encoder_input,
                decoder_labels,
                encoder_vocab_size,
                decoder_vocab_size,
                encoder_embedding_size=512,
                decoder_embedding_size=512,
                encoder_internal_size=512,
                decoder_internal_size=512,
                decoder_output_size=512,
                encoder_num_heads=8,
                decoder_num_heads=8,
                encoder_num_layers=6,
                decoder_num_layers=6,
                regularizer=None,
                dropout_rate=None,
                padding_id=0,
                start_of_sequence_id=2):
    with tf.variable_scope("transformer"):
        with tf.variable_scope("decoder_alignment"):
            decoder_input = tf.pad(decoder_labels, [[0, 0], [1, 0]],
                                   constant_values=start_of_sequence_id,
                                   name="decoder_input")
            decoder_labels = tf.pad(decoder_labels, [[0, 0], [0, 1]],
                                    constant_values=padding_id,
                                    name="decoder_labels")

        with tf.variable_scope("masking"):
            encoder_mask = padding_mask(encoder_input, padding_value=padding_id)
            decoder_mask = auto_regressive_mask(decoder_input)

        with tf.variable_scope("encoder_embedding"):
            encoder_embedding = positional_embedding(encoder_input,
                                                     encoder_vocab_size,
                                                     embedding_size=encoder_embedding_size,
                                                     regularizer=regularizer)

        with tf.variable_scope("decoder_embedding"):
            decoder_embedding = positional_embedding(decoder_input,
                                                     decoder_vocab_size,
                                                     embedding_size=decoder_embedding_size,
                                                     regularizer=regularizer)

        with tf.variable_scope("encoder_stack"):
            encoder_output = encoder_stack(encoder_embedding,
                                           internal_size=encoder_internal_size,
                                           num_heads=encoder_num_heads,
                                           num_layers=encoder_num_layers,
                                           regularizer=regularizer,
                                           dropout_rate=dropout_rate,
                                           mask=encoder_mask)

        with tf.variable_scope("decoder_stack"):
            decoder_output = decoder_stack(encoder_output,
                                           decoder_embedding,
                                           internal_size=decoder_internal_size,
                                           num_heads=decoder_num_heads,
                                           num_layers=decoder_num_layers,
                                           regularizer=regularizer,
                                           dropout_rate=dropout_rate,
                                           encoder_mask=encoder_mask,
                                           decoder_mask=decoder_mask)

        with tf.variable_scope("output"):
            weights = tf.get_variable(name="weights",
                                      shape=[decoder_output_size, decoder_vocab_size],
                                      initializer=tf.glorot_uniform_initializer(),
                                      regularizer=regularizer)
            logits = batch_matmul(decoder_output, weights)
            softmax = tf.nn.softmax(logits, name="softmax")

        with tf.variable_scope("accuracy"):
            mask = tf.cast(tf.not_equal(decoder_labels, padding_id), dtype=np.float32)
            argmax = tf.argmax(softmax, axis=-1, output_type=np.int32)
            accuracy = tf.cast(tf.equal(decoder_labels, argmax), dtype=np.float32)
            accuracy = tf.reduce_sum(accuracy * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)
            accuracy = tf.reduce_mean(accuracy)

        with tf.variable_scope("loss"):
            mask = tf.cast(tf.not_equal(decoder_input, padding_id), dtype=np.float32)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=decoder_labels)
            loss = tf.reduce_sum(loss * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)
            loss = tf.reduce_mean(loss)

        return softmax, loss, accuracy
