import unittest

import numpy as np
import tensorflow as tf

from activations import prelu
from attention import multi_head_attention
from embedding import positional_embedding
from encoder import encoder_layer, encoder_stack
from masking import auto_regressive_mask, padding_mask
from normalization import layer_normalization
from transformer import transformer


class TransformerBaseTest(unittest.TestCase):

    def setUp(self):
        tf.reset_default_graph()


class LayerNormalizationTest(TransformerBaseTest):

    def test_layer_norm(self):
        with tf.variable_scope("test_layer_norm"):
            inputs = tf.placeholder(dtype=np.float32, shape=(2, 3), name="inputs")

            layer_norm_output = layer_normalization(inputs, epsilon=1e-4)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            result = sess.run(feed_dict={
                inputs: np.array([[1, 2, 3], [4., 7., 19.]])
            }, fetches=[layer_norm_output,
                        "test_layer_norm/layer_normalization/mean:0",
                        "test_layer_norm/layer_normalization/stdev/Sqrt:0"])

            self.assertIsNotNone(result, "result is None")


class ActivationsTest(TransformerBaseTest):

    def test_prelu(self):
        with tf.variable_scope("test_prelu"):
            inputs = tf.placeholder(dtype=np.float32, shape=(2, 3), name="inputs")
            output = prelu(inputs, regularizer=tf.contrib.layers.l2_regularizer(0.01))

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            result = sess.run(feed_dict={
                inputs: np.array([[1, 2, 3], [-1, -2, -3]])
            }, fetches=[output])[0]

            expected = np.array([[1., 2., 3.], [-0.01, -0.02, -0.03]], dtype=np.float32)

            self.assertIsNotNone(result, "result is none")
            self.assertTrue(np.array_equal(result, expected), "result is incorrect")

            regularization_loss = sess.run(tf.losses.get_regularization_loss())
            self.assertEqual(regularization_loss, np.float32(1.4999999e-06), "l2 loss is incorrect")


class MultiHeadAttentionTest(TransformerBaseTest):

    def test_multi_head_attention(self):

        with tf.name_scope("test_multi_head_attention"):
            query_input = tf.placeholder(dtype=np.float32, shape=(None, 17, 99), name="query_input")
            key_input = tf.placeholder(dtype=np.float32, shape=(None, 25, 99), name="key_input")
            value_input = tf.placeholder(dtype=np.float32, shape=(None, 25, 145), name="value_input")
            dropout_input = tf.placeholder(dtype=np.float32, shape=(), name="dropout_input")

            attention = multi_head_attention([query_input, key_input, value_input])

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            result = sess.run(feed_dict={
                query_input: 1.0 * np.ones(shape=(3, 17, 99), dtype=np.float32),
                key_input: 2.0 * np.ones(shape=(3, 25, 99), dtype=np.float32),
                value_input: 3.0 * np.ones(shape=(3, 25, 145), dtype=np.float32),
                dropout_input: np.float32(0.2)
            }, fetches=[attention])[0]

            self.assertIsNotNone(result)

    def test_padding_mask(self):

        with tf.variable_scope("test_padding_mask"):
            inputs = tf.placeholder(dtype=np.int32, shape=(None, 5), name="token_input")

            mask = padding_mask(inputs)

            embedding = positional_embedding(inputs=inputs,
                                             vocab_size=10000,
                                             embedding_size=256,
                                             regularizer=tf.contrib.layers.l2_regularizer(0.01))

            attention = multi_head_attention(inputs=[embedding, embedding, embedding], mask=mask)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            result = sess.run(feed_dict={
                inputs: np.array([[1, 2, 0, 0, 0],
                                  [1, 2, 3, 4, 0]], dtype=np.int32),
            }, fetches=[attention, "test_padding_mask/multi_head_attention/softmax:0"])

            self.assertIsNotNone(result, "result is None")

            attention_weights = result[1]

            self.assertEqual(attention_weights.shape, (2, 8, 5, 5), "attention shape is incorrect")
            self.assertTrue(np.array_equal(attention_weights[0][0][:, 2:], np.zeros((5, 3))),
                            "attention mask is incorrect")
            self.assertTrue(np.array_equal(attention_weights[1][0][:, 4:], np.zeros((5, 1))),
                            "attention mask is incorrect")

            self.assertEqual(np.sum(attention_weights[0][0][0][2:]), 0.,
                             "softmax attention {} is incorrect".format(attention_weights[0][0][0]))
            self.assertTrue(abs(np.sum(attention_weights[0][0][0][:2]) - 1.) < 2e-6,
                            "softmax attention {} is incorrect".format(attention_weights[0][0][0]))
            self.assertTrue(abs(np.sum(attention_weights) - 80.) < 2e-8,
                            "softmax attention {} is incorrect".format(attention_weights))

    def test_auto_regressive_mask(self):

        with tf.variable_scope("test_auto_regressive_mask"):
            inputs = tf.placeholder(dtype=np.int32, shape=(None, 5), name="token_input")

            mask = auto_regressive_mask(inputs)

            embedding = positional_embedding(inputs=inputs,
                                             vocab_size=10000,
                                             embedding_size=256,
                                             regularizer=tf.contrib.layers.l2_regularizer(0.01))

            attention = multi_head_attention(inputs=[embedding, embedding, embedding], mask=mask)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            result = sess.run(feed_dict={
                inputs: np.array([[1, 2, 0, 0, 0],
                                  [1, 2, 3, 4, 0]], dtype=np.int32),
            }, fetches=[attention, "test_auto_regressive_mask/multi_head_attention/softmax:0"])

            self.assertIsNotNone(result, "result is None")

            attention_weights = result[1]

            for i in range(len(attention_weights)):
                for j in range(len(attention_weights[i])):
                    for k in range(len(attention_weights[i][j])):
                        self.assertTrue(abs(np.sum(attention_weights[i][j][k]) - 1.) < 2e-6)
                        self.assertTrue(abs(np.sum(attention_weights[i][j][k][:k + 1]) - 1.) < 2e-6)
                        self.assertEqual(np.sum(attention_weights[i][j][k][k + 1:]), 0,
                                         "softmax attention {} is incorrect".format(attention_weights))


class EncoderTest(TransformerBaseTest):

    def test_layer(self):
        inputs = tf.placeholder(np.int32, shape=[None, 32], name="inputs")

        embedding = positional_embedding(inputs, 2000,
                                         embedding_size=256,
                                         regularizer=tf.contrib.layers.l2_regularizer(0.01))

        encoder = encoder_layer(embedding,
                                dropout_rate=tf.placeholder_with_default(tf.constant(0.), [], name="dropout_rate"),
                                internal_size=256,
                                num_heads=8,
                                regularizer=tf.contrib.layers.l2_regularizer(0.01))

        self.assertEquals(encoder.shape.as_list(), [None, 32, 256])

    def test_stack(self):
        inputs = tf.placeholder(np.int32, shape=[None, 32], name="inputs")

        embedding = positional_embedding(inputs, 2000,
                                         embedding_size=256,
                                         regularizer=tf.contrib.layers.l2_regularizer(0.01))

        mask = padding_mask(inputs)

        encoder = encoder_stack(embedding,
                                dropout_rate=tf.placeholder_with_default(tf.constant(0.0), name="dropout_rate",
                                                                         shape=[]),
                                internal_size=256,
                                mask=mask,
                                num_heads=8,
                                num_layers=4,
                                regularizer=tf.contrib.layers.l2_regularizer(0.01))

        self.assertEquals(encoder.shape.as_list(), [None, 32, 256])


class DecoderTest(TransformerBaseTest):

    def test_alignment(self):
        inputs = tf.placeholder(np.int32, shape=[None, 4], name="inputs")

        outputs = tf.pad(inputs, [[0, 0], [1, 0]], constant_values=9)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        result = sess.run(feed_dict={
            inputs: np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8]], dtype=np.int32),
        }, fetches=[outputs])

        self.assertIsNotNone(result)


class TransformerTest(TransformerBaseTest):

    def test_simple_training(self):
        encoder_input = tf.placeholder(np.int32, shape=[None, 5], name="encoder_input")
        decoder_labels = tf.placeholder(np.int32, shape=[None, 6], name="decoder_labels")

        softmax, loss, accuracy = transformer(encoder_input,
                                              decoder_labels,
                                              encoder_vocab_size=6,
                                              decoder_vocab_size=10)

        # encoder
        # ------------------------
        # 0: padding
        # 1: unknown
        encoder_input_data = [[2, 3, 4, 0, 0],
                              [5, 4, 3, 2, 0],
                              [2, 3, 4, 3, 2]]

        # decoder
        # -------------------------
        # 0: padding
        # 1: unknown
        # 2: start of sequence
        # 3: end of sequence
        decoder_input_data = [[9, 8, 7, 3, 0, 0],
                              [4, 5, 6, 7, 8, 3],
                              [9, 8, 3, 0, 0, 0]]

        optimizer = tf.train.AdamOptimizer().minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        result = sess.run(
            feed_dict={
                encoder_input: encoder_input_data,
                decoder_labels: decoder_input_data
            }, fetches=[softmax, loss, accuracy,
                        "transformer/decoder_alignment/decoder_labels:0",
                        "transformer/decoder_alignment/decoder_input:0",
                        "transformer/accuracy/ArgMax:0"])

        self.assertEqual(result[0].shape, tuple([3, 7, 10]))

        for i in range(200):
            result = sess.run(
                feed_dict={
                    encoder_input: encoder_input_data,
                    decoder_labels: decoder_input_data
                }, fetches=[optimizer, softmax, loss, accuracy])
            print(f"{i:<5} {result[2]} {result[3]}")

    def test_loss(self):
        labels = tf.placeholder(np.int32, shape=[None, 4], name="labels")
        logits = tf.placeholder(np.float32, shape=[None, 4, 3], name="logits")

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        mask = tf.cast(tf.not_equal(labels, 0), np.float32)
        loss = tf.reduce_sum(loss * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)
        loss = tf.reduce_mean(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        result = sess.run(feed_dict={
            labels: np.array([[1, 2, 0, 0],
                              [2, 2, 1, 0]], dtype=np.int32),

            logits: np.array([[[1.5, 2.9, 1.1],
                               [0.5, 1.9, 4.6],
                               [0.1, 3.2, 2.7],
                               [9.5, 0.2, 3.4]],

                              [[1.1, 2.9, 5.1],
                               [0.7, 3.9, 1.6],
                               [0.1, 0.2, 0.7],
                               [0.5, 0.2, 0.4]]], dtype=np.float32)
        }, fetches=[loss, mask])

        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
