# Transformer

# Getting Started
Start by defining the encoder inputs and decoder labels as `tf.placeholders`.

```python
encoder_input = tf.placeholder(np.int32, shape=[None, 5], name="encoder_input")
decoder_labels = tf.placeholder(np.int32, shape=[None, 6], name="decoder_labels")
```
Instantiate a Transformer with the placeholders and options:

```python
softmax, loss, accuracy = transformer(encoder_input,
                                      decoder_labels,
                                      encoder_vocab_size=6,
                                      decoder_vocab_size=10)
```
### Example 
Setup some example data:

```python
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
```
### Execution
Execute the training loop as follows:

```python
optimizer = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(200):
    result = sess.run(
        feed_dict={
            encoder_input: encoder_input_data,
            decoder_labels: decoder_input_data
        }, fetches=[optimizer, softmax, loss, accuracy])
    print(f"{i:<5} {result[2]} {result[3]}")
```
## Conclusion
I invite feedback from the AI and Machine Learning communities. 
