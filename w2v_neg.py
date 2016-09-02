import math
import pickle
import argparse
import numpy as np
import logging as log
import tensorflow as tf


def read_dict(file_name):
    with open(file_name, "rb") as fd:
        return pickle.load(fd)


def build_input_pipeline(input_file, batch_size):
    input_files = tf.train.string_input_producer([input_file])
    reader = tf.FixedLengthRecordReader(record_bytes=4+4)
    _, raw_val_t = reader.read(input_files)
    int_val_t = tf.decode_raw(raw_val_t, tf.int32)
    center_t, context_t = int_val_t[0], int_val_t[1]

    center_batch_t, context_batch_t = tf.train.shuffle_batch([center_t, context_t], batch_size,
                                                             min_after_dequeue=10*batch_size,
                                                             capacity=20*batch_size,
                                                             num_threads=2)
    return center_batch_t, context_batch_t


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
    num_sampled = 10
    log.info("Training params: vec_size=%d, batch=%d, num_neg=%d", vec_size, batch_size, num_sampled)

    embeddings = tf.Variable(tf.random_uniform([dict_size, vec_size], -1.0, 1.0))
    nce_weights = tf.Variable(
        tf.truncated_normal([dict_size, vec_size],
                            stddev=1.0 / math.sqrt(vec_size)))

    center_batch_t, context_batch_t = build_input_pipeline(args.train, batch_size)
    # we need to reshape context to add 1 dimension
    context_batch_t = tf.expand_dims(context_batch_t, 1)

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

    global_step_t = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_t, global_step=global_step_t)

    init = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord)

        try:
            while True:
                batch, labels = session.run([center_batch_t, context_batch_t])

                feed_dict = {
                    train_inputs_t: batch,
                    train_labels_t: labels
                }

                _, step, loss = session.run([optimizer, global_step_t, loss_t], feed_dict=feed_dict)

                if step % 100 == 0:
                    print(step, loss)
        finally:
            coord.request_stop()

        coord.join(threads)

