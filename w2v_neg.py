import math
import pickle
import argparse
import numpy as np
import logging as log
import tensorflow as tf


def read_dict(file_name):
    with open(file_name, "rb") as fd:
        return pickle.load(fd)


def build_input_pipeline(input_prefix):
    ctr_input_files = tf.train.string_input_producer([input_prefix + ".center"])
    ctx_input_files = tf.train.string_input_producer([input_prefix + ".context"])

    ctr_reader = tf.FixedLengthRecordReader()


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dict", required=True, help="File with dict serialized data")
    parser.add_argument("-t", "--train", required=True, help="Train data file")
    args = parser.parse_args()

    log.info("Reading dict from %s", args.dict)
    dict_data = read_dict(args.dict)
    log.info("Dict has %d entries", len(dict_data))

    dict_size = len(dict_data)
    vec_size = 100
    batch_size = 128
    num_sampled = 20
    log.info("Training params: vec_size=%d, batch=%d, num_neg=%d", vec_size, batch_size, num_sampled)

    embeddings = tf.Variable(tf.random_uniform([dict_size, vec_size], -1.0, 1.0))
    nce_weights = tf.Variable(
        tf.truncated_normal([dict_size, vec_size],
                            stddev=1.0 / math.sqrt(vec_size)))

    build_input_pipeline(args.train)

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

        for iter_idx, (batch, labels) in enumerate(generate_batches(dataset, batch_size, context_size_one)):
            feed_dict = {
                train_inputs_t: batch,
                train_labels_t: labels
            }

            _, loss = session.run([optimizer, loss_t], feed_dict=feed_dict)

            if iter_idx % 100 == 0:
                print(iter_idx, loss)

            
