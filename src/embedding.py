import numpy as np
import tensorflow as tf


def positional_signal(sequence_length: int = None,
                      embedding_size: int = None,
                      min_timescale: float = 1.,
                      max_timescale: float = 1.e4):
    with tf.variable_scope("positional_signal"):
        positions = np.arange(sequence_length, dtype=np.float32)
        timescale_size = embedding_size / 2.
        timescale_increment = np.log(max_timescale / min_timescale) / timescale_size
        inverse_timescale = min_timescale * np.exp(np.arange(timescale_size, dtype=np.float32) * -timescale_increment)
        timescale = np.multiply(np.expand_dims(positions, 1), np.expand_dims(inverse_timescale, 0))
        signal = np.hstack([np.sin(timescale), np.cos(timescale)])
        return tf.expand_dims(signal, axis=0)


def positional_embedding(
        inputs,
        vocab_size,
        embedding_size=512,
        regularizer=None,
        min_timescale=1.,
        max_timescale=1.e4):
    if embedding_size % 2:
        raise ValueError("embedding_size should be even")

    sequence_length = inputs.shape[1].value

    with tf.variable_scope("positional_embedding", reuse=tf.AUTO_REUSE):
        embedding_weights = tf.get_variable(name="embedding_weights",
                                            shape=[vocab_size, embedding_size],
                                            initializer=tf.random_uniform_initializer(),
                                            regularizer=regularizer)

        embedding_lookup = tf.nn.embedding_lookup(embedding_weights, inputs)

        signal = positional_signal(sequence_length=sequence_length,
                                   embedding_size=embedding_size,
                                   min_timescale=min_timescale,
                                   max_timescale=max_timescale)

        return embedding_lookup + signal
