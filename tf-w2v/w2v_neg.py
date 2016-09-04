#!/usr/bin/env python3

import glob
import os
import math
import pickle
import argparse
import logging as log
import numpy as np
import tensorflow as tf

from time import time
from datetime import timedelta


def read_dict(file_name):
    with open(file_name, "rb") as fd:
        return pickle.load(fd)


def build_input_pipeline(input_file, batch_size):
    input_files = tf.train.string_input_producer([input_file])
    reader = tf.FixedLengthRecordReader(record_bytes=4+4)
    _, raw_val_t = reader.read(input_files)
    int_val_t = tf.decode_raw(raw_val_t, tf.int32)
    center_t, context_t = int_val_t[0], int_val_t[1]

    center_batch_t, context_batch_t = tf.train.batch([center_t, context_t], batch_size, num_threads=4, capacity=1024)
    return center_batch_t, context_batch_t


def get_num_entries(input_file):
    st = os.stat(input_file)
    return st.st_size / 8


if __name__ == "__main__":
    REPORT_EVERY_STEPS = 100
    SAVE_EMBEDDINGS_EVERY = 1000

    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dict", required=True, help="File with dict serialized data")
    parser.add_argument("-t", "--train", required=True, help="Train data file")
    parser.add_argument("-r", "--run", default="w2v", help="Name of this run, default=w2v")
    args = parser.parse_args()

    log.info("Reading dict from %s", args.dict)
    dict_data = read_dict(args.dict)
    log.info("Dict has %d entries", len(dict_data))

    input_entries = get_num_entries(args.train)
    log.info("Input data has %d samples", input_entries)

    dict_size = len(dict_data)
    vec_size = 100
    batch_size = 1024*10
    num_sampled = 10
    log.info("Training params: vec_size=%d, batch=%d, num_neg=%d", vec_size, batch_size, num_sampled)

    embeddings = tf.Variable(tf.random_uniform([dict_size, vec_size], -1.0, 1.0))

    norm_t = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings_t = embeddings / norm_t

    nce_weights = tf.Variable(
        tf.truncated_normal([dict_size, vec_size],
                            stddev=1.0 / math.sqrt(vec_size)))

    center_batch_t, context_batch_t = build_input_pipeline(args.train, batch_size)

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
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.03).minimize(loss_t, global_step=global_step_t)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss_t, global_step=global_step_t)

    # summaries
    tf.scalar_summary('loss_cur', loss_t)
    loss_avg_t = tf.Variable(0.0, name='loss_avg', trainable=False)
    tf.scalar_summary('loss_avg', loss_avg_t)
    speed_t = tf.Variable(0.0, name="speed", trainable=False)
    tf.scalar_summary('speed', speed_t)

    init = tf.initialize_all_variables()
    merged_summaries = tf.merge_all_summaries()
    time_started = time()

    # limit amount of allocated memory to 1/2 of 8GB
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        summary_writer = tf.train.SummaryWriter("logs/" + args.run, session.graph)
        session.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(session, coord)

        try:
            sum_loss = 0.0
            while True:
                batch, labels = session.run([center_batch_t, context_batch_t])

                feed_dict = {
                    train_inputs_t: batch,
                    train_labels_t: np.expand_dims(labels, 1)
                }

                _, step, loss = session.run([optimizer, global_step_t, loss_t], feed_dict=feed_dict)
                sum_loss += loss

                if step % REPORT_EVERY_STEPS == 0:
                    epoch = int(step * batch_size / input_entries)
                    sec_passed = time() - time_started
                    speed = step * batch_size / sec_passed

                    summary_res, = session.run([merged_summaries], feed_dict={
                        loss_avg_t: sum_loss / REPORT_EVERY_STEPS,
                        loss_t: loss,
                        speed_t: speed,
                    })
                    summary_writer.add_summary(summary_res, step)

                    print("%s: epoch=%d, step=%d, loss=%f, speed=%.3f samples/s" % (
                        timedelta(seconds=sec_passed), epoch,
                        step, sum_loss / REPORT_EVERY_STEPS, speed))
                    sum_loss = 0.0

                if step % SAVE_EMBEDDINGS_EVERY == 0:
                    norm_embeddings, = session.run([normalized_embeddings_t])
                    existing = glob.glob(args.run + "-*.npy")
                    np.save(args.run + "-%d" % step, norm_embeddings)
                    for prev_save in existing:
                        os.unlink(prev_save)
        finally:
            coord.request_stop()

        coord.join(threads)

