import tensorflow as tf

from common import batch_matmul


def combine_heads(inputs, internal_size):
    with tf.name_scope("combine_heads"):
        sequence_length = inputs.shape[2].value
        inputs = tf.transpose(inputs, [0, 2, 1, 3])
        return tf.reshape(inputs, [-1, sequence_length, internal_size])


def split_heads(inputs, num_heads, depth):
    with tf.name_scope("split_heads"):
        sequence_length = inputs.shape[1].value
        inputs = tf.reshape(inputs, [-1, sequence_length, num_heads, depth])
        return tf.transpose(inputs, [0, 2, 1, 3])


def multi_head_attention(
        inputs,
        internal_size=512,
        output_size=512,
        num_heads=8,
        regularizer=None,
        mask=None):
    if internal_size % num_heads:
        raise ValueError("internal size {} % num heads {} != 0".format(internal_size, num_heads))

    with tf.variable_scope("multi_head_attention"):

        depth = internal_size // num_heads

        scaling_factor = depth ** -0.5

        query_input = inputs[0]
        key_input = inputs[1]
        value_input = inputs[2]

        query_columns = query_input.shape[2]
        key_columns = key_input.shape[2]
        value_columns = value_input.shape[2]

        if query_columns != key_columns:
            raise ValueError("query columns {} != key columns {}".format(query_columns, key_columns))

        key_rows = inputs[1].shape[1]
        value_rows = inputs[2].shape[1]

        if key_rows != value_rows:
            raise ValueError("query rows {} != key rows {}".format(key_rows, value_rows))

        query_weights = tf.get_variable(
            name="query_weights",
            shape=[query_columns, internal_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=regularizer)

        key_weights = tf.get_variable(
            name="key_weights",
            shape=[key_columns, internal_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=regularizer)

        value_weights = tf.get_variable(
            name="value_weights",
            shape=[value_columns, internal_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=regularizer)

        output_weights = tf.get_variable(
            name="output_weights",
            shape=[internal_size, output_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=regularizer)

        with tf.variable_scope("query_projection"):
            query = batch_matmul(query_input, query_weights)

        with tf.variable_scope("key_projection"):
            key = batch_matmul(key_input, key_weights)

        with tf.variable_scope("value_projection"):
            value = batch_matmul(value_input, value_weights)

        with tf.variable_scope("split_query"):
            split_query = split_heads(query, num_heads, depth)

        with tf.variable_scope("split_key"):
            split_key = split_heads(key, num_heads, depth)

        with tf.variable_scope("split_value"):
            split_value = split_heads(value, num_heads, depth)

        with tf.variable_scope("query_key_mixer"):
            split_query_key = tf.matmul(split_query, split_key, transpose_b=True)

        if mask is not None:
            with tf.name_scope("masking"):
                split_query_key += mask

        split_softmax = tf.nn.softmax(scaling_factor * split_query_key, name="softmax")

        with tf.variable_scope("apply_attention"):
            split_output = tf.matmul(split_softmax, split_value)

        combined_output = combine_heads(split_output, internal_size)

        with tf.variable_scope("output_projection"):
            return batch_matmul(combined_output, output_weights)
