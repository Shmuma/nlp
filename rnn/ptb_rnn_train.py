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

from lib import ptb, vocab, rnn

BATCH = 64
NUM_STEPS = 10
EMBEDDING = 50
CELL_SIZE = 100
LR = 0.001
DROPOUT = 0.9

LOG_DIR = "logs"
SAVE_DIR = "saves"


def make_net(vocab_size, num_steps=NUM_STEPS, batch=BATCH, embeddings=None):
    ph_input = tf.placeholder(tf.int32, shape=(None, num_steps), name="input")
    ph_dropout = tf.placeholder(tf.float32, name='Dropout')

    with tf.variable_scope("Net"):
        cell = tf.nn.rnn_cell.BasicRNNCell(CELL_SIZE, activation=tf.sigmoid)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=ph_dropout, output_keep_prob=ph_dropout)
        # that's weird, but using xavier initializer stable gives -10..-15 to final perplexity
        # maybe, something worth to investigate
        if embeddings is None:
            cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, vocab_size, EMBEDDING,
                                                   initializer=tf.contrib.layers.xavier_initializer())
        else:
            cell = rnn.EmbeddingFastTextWrapper(cell, embeddings)
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, vocab_size)
        initial_state = cell.zero_state(batch, dtype=tf.float32)

        inputs = [tf.squeeze(val, squeeze_dims=[1]) for val in tf.split(split_dim=1, num_split=num_steps, value=ph_input)]
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=initial_state)
    return ph_input, initial_state, outputs, state, ph_dropout


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="rnn", help="Name of the run used in saving logs and models")
    parser.add_argument("--max-epoch", type=int, default=16,
                        help="If specified, stop after given amount of epoches, default=16")
    parser.add_argument("-e", "--embeddings", help="File name with embeddings to be used, if not specified, "
                                                   "embeddings will be trained")
    args = parser.parse_args()

    os.makedirs(os.path.join(LOG_DIR, args.name), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, args.name), exist_ok=True)

    log.info("Loading PTB dataset...")
    if args.embeddings:
        v = vocab.FastTextVocab(args.embeddings)
    else:
        v = vocab.Vocab()
    data = ptb.PTBDataset("data", v, batch_size=BATCH)
    data.load_dataset()
    log.info("Loaded: %s", data)

    with tf.Session() as session:
        ph_labels = tf.placeholder(tf.int32, shape=(None, NUM_STEPS), name="labels")

        ph_input, initial_state, outputs, final_state, ph_dropout = make_net(data.vocab.size(),
                                                                             embeddings=args.embeddings)

#        targets = tf.split(split_dim=1, num_split=NUM_STEPS, value=ph_labels)
        output = tf.reshape(tf.concat(1, outputs), [-1, data.vocab.size()])
        log.info("Loss info:")
        log.info("Output: %s", output)
        labels = tf.reshape(ph_labels, [-1])
        log.info("Labels: %s", labels)
        loss_t = sequence_loss([output], [labels], [tf.ones([BATCH * NUM_STEPS])])

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
        init_state = initial_state.eval()

        global_step = 0
        epoch = 0
        while args.max_epoch is None or args.max_epoch > epoch:
            losses = []
            state = init_state
            for train_x, train_y, progress in data.iterate_train(NUM_STEPS):
                loss, state, _ = session.run([loss_t, final_state, opt_t], feed_dict={
                    ph_input: train_x,
                    ph_labels: train_y,
                    initial_state: state,
                    ph_dropout: DROPOUT
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
            saver.save(session, os.path.join(SAVE_DIR, args.name, "model"), global_step=epoch)

            # validation
            log.info("Running validation...")
            losses = []
            state = init_state
            for x, y, _ in data.iterate_validation(NUM_STEPS):
                loss, state, res_outs = session.run([loss_t, final_state, outputs], feed_dict={
                    ph_input: x,
                    ph_labels: y,
                    initial_state: state,
                    ph_dropout: 1.0
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
        state = init_state
        for x, y, _ in data.iterate_test(NUM_STEPS):
            loss, state = session.run([loss_t, final_state], feed_dict={
                ph_input: x,
                ph_labels: y,
                initial_state: state,
                ph_dropout: 1.0
            })
            losses.append(loss)
        log.info("Test perplexity: %s", np.exp(np.mean(losses)))

    pass
