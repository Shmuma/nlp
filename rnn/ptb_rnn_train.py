#!/usr/bin/env python
"""
Trains simple language model using RNN and PTB dataset
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
import logging as log

import sys
sys.path.append("..")

from lib import ptb, vocab

BATCH = 64
NUM_STEPS = 10
EMBEDDING = 100
CELL_SIZE = 100
LR = 0.001

LOG_DIR = "logs"
SAVE_DIR = "saves"


def make_net(ph_input, vocab_size, num_steps=NUM_STEPS, batch=BATCH):
    cell = tf.nn.rnn_cell.BasicRNNCell(CELL_SIZE)
    cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, vocab_size)
    initial_state = cell.zero_state(batch, dtype=tf.float32)

    with tf.variable_scope("W2V"):
        embedding = tf.get_variable("embedding", [vocab_size, EMBEDDING])
        inputs = tf.nn.embedding_lookup(embedding, ph_input)
        inputs = [tf.squeeze(val, squeeze_dims=[1]) for val in tf.split(split_dim=1, num_split=num_steps, value=inputs)]
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=initial_state)
    return initial_state, outputs, state


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="rnn", help="Name of the run used in saving logs and models")
    parser.add_argument("--max-epoch", type=int, default=16,
                        help="If specified, stop after given amount of epoches, default=16")
    args = parser.parse_args()

    os.makedirs(os.path.join(LOG_DIR, args.name), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, args.name), exist_ok=True)

    log.info("Loading PTB dataset...")
    data = ptb.PTBDataset("data", vocab.Vocab(), num_steps=10)
    data.load_train()
    log.info("Loaded: %s", data)

    with tf.Session() as session:
        ph_input = tf.placeholder(tf.int32, shape=(None, NUM_STEPS), name="input")
        ph_labels = tf.placeholder(tf.int32, shape=(None, NUM_STEPS), name="labels")

        initial_state, outputs, final_state = make_net(ph_input, data.vocab.size())

        targets = tf.split(split_dim=1, num_split=NUM_STEPS, value=ph_labels)
        loss_t = sequence_loss(outputs, targets, [tf.constant(1.0, tf.float32) for _ in range(NUM_STEPS)])
        opt = tf.train.AdamOptimizer(LR)
        opt_t = opt.minimize(loss_t)

        # summaries
        writer = tf.train.SummaryWriter(os.path.join(LOG_DIR, args.name), session.graph)
        summary_loss_ph = tf.placeholder(tf.float32, name='loss')
        tf.scalar_summary("loss", summary_loss_ph)

        saver = tf.train.Saver()

        summaries = tf.merge_all_summaries()
        session.run(tf.initialize_all_variables())
        losses = []

        global_step = 0
        epoch = 0
        while args.max_epoch is None or args.max_epoch > epoch:
            for iter_no, (train_x, train_y, progress) in enumerate(data.iterate_train(BATCH)):
                loss, _ = session.run([loss_t, opt_t], feed_dict={
                    ph_input: train_x,
                    ph_labels: train_y
                })
                losses.append(loss)
                if iter_no % 100 == 0:
                    m_loss = np.mean(losses)
                    log.info("Epoch=%d, iter=%d, epoch_perc=%.2f%%, mean_loss=%s",
                             epoch, iter_no, progress*100.0, m_loss)
                    summ_res, = session.run([summaries], feed_dict={
                        summary_loss_ph: m_loss,
                    })
                    writer.add_summary(summ_res, global_step)
                    writer.flush()
                    losses = []
                global_step += 1
            saver.save(session, os.path.join(SAVE_DIR, args.name, "model-epoch=%d" % epoch))
            epoch += 1
    pass
