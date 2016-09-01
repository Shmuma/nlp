import tensorflow as tf
import math
import numpy as np

import sys
import collections

from cs224d.data_utils import StanfordSentiment


def generate_batches(dataset, batch_size, context_one_side):
    """
    Infinitely yield batches for given size and epoch number
    """
    assert isinstance(dataset, StanfordSentiment)

    batch = np.ndarray(shape=(batch_size, ), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    tokens = dataset.tokens()
    ofs = 0

    while True:
        center, ctx_words = dataset.getRandomContext(context_one_side)
        c_id = tokens[center]
        for ctx_word in ctx_words:
            ctx_id = tokens[ctx_word]
            batch[ofs] = c_id
            labels[ofs] = ctx_id
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
    context_size_one = 3

    embeddings = tf.Variable(tf.random_uniform([dict_size, vec_size], -1.0, 1.0))
    nce_weights = tf.Variable(
        tf.truncated_normal([dict_size, vec_size],
                            stddev=1.0 / math.sqrt(vec_size)))

    # Placeholders for inputs
    train_inputs_t = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels_t = tf.placeholder(tf.int32, shape=[batch_size, 1])
    embed = tf.nn.embedding_lookup(embeddings, train_inputs_t)
    nce_weights = tf.Variable(
        tf.truncated_normal([dict_size, vec_size],
                            stddev=1.0 / math.sqrt(vec_size)))
    nce_biases = tf.Variable(tf.zeros([dict_size]))

    loss_t = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels_t,
                       num_sampled, dict_size))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss_t)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        init.run()
        print("Initialized")
        
        for iter_idx, (batch, labels) in enumerate(generate_batches(dataset, batch_size, context_size_one)):
            feed_dict = {
                train_inputs_t: batch,
                train_labels_t: labels
            }

            _, loss = session.run([optimizer, loss_t], feed_dict=feed_dict)

            if iter_idx % 100 == 0:
                print(iter_idx, loss)

            
