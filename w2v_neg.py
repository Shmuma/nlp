import tensorflow as tf
import math
import numpy as np

import sys
import collections

from cs224d.data_utils import StanfordSentiment


def generate_batches(dataset, batch_size, num_skips, skip_window):
    """
    Infinitely yield batches for given size and epoch number
    """
    assert isinstance(dataset, StanfordSentiment)

    batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,), dtype=np.int32)
    tokens = dataset.tokens()
    ofs = 0

    while True:
        center, ctx_words = dataset.getRandomContext(num_skips)
        c_id = tokens[center]
        for ctx_word in ctx_words:
            ctx_id = tokens[ctx_word]
            batch[ofs] = ctx_id
            labels[ofs] = c_id
            ofs += 1
            if ofs == batch_size:
                yield batch, labels
                ofs = 0

if __name__ == "__main__":
    print("Initialize dataset")
    dataset = StanfordSentiment()
    print("Dictionary size=%d" % len(dataset.tokens()))

    dict_size = len(dataset.tokens())
    vec_size = 100
    batch_size = 50
    num_sampled = 20
    max_steps = 10

    embeddings = tf.Variable(tf.random_uniform([dict_size, vec_size], -1.0, 1.0))
    nce_weights = tf.Variable(
        tf.truncated_normal([dict_size, vec_size],
                            stddev=1.0 / math.sqrt(vec_size)))

    # Placeholders for inputs
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    nce_weights = tf.Variable(
        tf.truncated_normal([dict_size, vec_size],
                            stddev=1.0 / math.sqrt(vec_size)))
    nce_biases = tf.Variable(tf.zeros([dict_size]))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                       num_sampled, dict_size))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        init.run()
        print("Initialized")
        
        generate_batches(dataset, 5, 2, 3)

            
