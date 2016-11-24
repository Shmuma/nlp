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

from lib.utils import calculate_perplexity, get_ptb_dataset, Vocab
from lib.utils import ptb_iterator, sample


BATCH = 64
NUM_STEPS = 10
EMBEDDING = 50
CELL_SIZE = 100
LR = 0.001
DROPOUT = 0.9

LOG_DIR = "logs"
SAVE_DIR = "saves"


def make_net(ph_input, vocab_size, dropout_prob=DROPOUT, num_steps=NUM_STEPS, batch=BATCH):
    with tf.variable_scope("Net", initializer=tf.contrib.layers.xavier_initializer()):
        cell = tf.nn.rnn_cell.BasicRNNCell(CELL_SIZE, activation=tf.sigmoid)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_prob, output_keep_prob=dropout_prob)
        cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, vocab_size, EMBEDDING)
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, vocab_size)
        initial_state = cell.zero_state(batch, dtype=tf.float32)

        # with tf.variable_scope("W2V"):
        #     embedding = tf.get_variable("embedding", [vocab_size, EMBEDDING])
        #     inputs = tf.nn.embedding_lookup(embedding, ph_input)
        #     inputs = [tf.squeeze(val, squeeze_dims=[1]) for val in tf.split(split_dim=1, num_split=num_steps, value=inputs)]
        inputs = [tf.squeeze(val, squeeze_dims=[1]) for val in tf.split(split_dim=1, num_split=num_steps, value=ph_input)]
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=initial_state)
    return initial_state, outputs, state


class Data:
    def __init__(self):
        self.vocab = Vocab()
        self.vocab.construct(get_ptb_dataset('train'))
        self.encoded_train = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('train')],
            dtype=np.int32)
        self.encoded_valid = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
            dtype=np.int32)
        self.encoded_test = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('test')],
            dtype=np.int32)


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
#    data = ptb.PTBDataset("data", vocab.Vocab(), num_steps=10)
#    data.load_dataset()
#    log.info("Loaded: %s", data)
    data = Data()

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
        summ_perpl_train_t = tf.placeholder(tf.float32, name='perplexity_train')
        tf.scalar_summary("perplexity_train", summ_perpl_train_t)

        summ_perpl_val_t = tf.placeholder(tf.float32, name='perplexity_val')
        tf.scalar_summary("perplexity_val", summ_perpl_val_t, collections=['summary_epoch'])

        saver = tf.train.Saver(max_to_keep=args.max_epoch)

        summaries = tf.merge_all_summaries()
        summaries_epoch = tf.merge_all_summaries('summary_epoch')
        session.run(tf.initialize_all_variables())

        global_step = 0
        epoch = 0
        progress = 0.0
        while args.max_epoch is None or args.max_epoch > epoch:
            losses = []
            state = initial_state.eval()
            for train_x, train_y in ptb_iterator(data.encoded_train, BATCH, NUM_STEPS):
                loss, state, _ = session.run([loss_t, final_state, opt_t], feed_dict={
                    ph_input: train_x,
                    ph_labels: train_y,
                    initial_state: state
                })
                losses.append(loss)
                if global_step % 100 == 0:
                    m_perpl = np.exp(np.mean(losses))
                    log.info("Epoch=%d, iter=%d, epoch_perc=%.2f%%, perplexity=%s",
                             epoch, global_step, progress*100.0, m_perpl)
                    summ_res, = session.run([summaries], feed_dict={
                        summ_perpl_train_t: m_perpl,
                    })
                    writer.add_summary(summ_res, global_step)
                    writer.flush()
                    losses = []
                global_step += 1
#            saver.save(session, os.path.join(SAVE_DIR, args.name, "model-epoch=%d" % epoch))

            # validation
            log.info("Running validation...")
            losses = []
            for x, y in ptb_iterator(data.encoded_valid, BATCH, NUM_STEPS):
                loss, = session.run([loss_t], feed_dict={
                    ph_input: x,
                    ph_labels: y
                })
                losses.append(loss)
            m_perpl = np.exp(np.mean(losses))
            summ_res, = session.run([summaries_epoch], feed_dict={
                summ_perpl_val_t: m_perpl
            })
            writer.add_summary(summ_res, global_step)
            writer.flush()
            log.info("Validiation perplexity: %s", m_perpl)
            epoch += 1

        log.info("Running test...")
        losses = []
        for x, y in ptb_iterator(data.encoded_test, BATCH, NUM_STEPS):
            loss, = session.run([loss_t], feed_dict={
                ph_input: x,
                ph_labels: y
            })
            losses.append(loss)
        log.info("Test perplexity: %s", np.exp(np.mean(losses)))

    pass
