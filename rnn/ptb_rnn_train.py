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

if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading PTB dataset...")
    data = ptb.PTBDataset("data", vocab.Vocab(), num_steps=10)
    log.info("Loaded: %s", data)

    with tf.Session() as session:
        ph_input = tf.placeholder(tf.int32, shape=(None, NUM_STEPS), name="input")
        ph_labels = tf.placeholder(tf.int32, shape=(None, NUM_STEPS), name="labels")
        cell = tf.nn.rnn_cell.BasicRNNCell(CELL_SIZE)
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, data.vocab.size())
        initial_state = cell.zero_state(BATCH, dtype=tf.float32)

        with tf.variable_scope("Model"):
            embedding = tf.get_variable("embedding", [data.vocab.size(), EMBEDDING])
            inputs = tf.nn.embedding_lookup(embedding, ph_input)
            inputs = [tf.squeeze(val, squeeze_dims=[1]) for val in tf.split(split_dim=1, num_split=NUM_STEPS, value=inputs)]
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=initial_state)

        targets = tf.split(split_dim=1, num_split=NUM_STEPS, value=ph_labels)
        loss_t = sequence_loss(outputs, targets, [tf.constant(1.0, tf.float32) for _ in range(NUM_STEPS)])
        opt = tf.train.AdamOptimizer(LR)
        opt_t = opt.minimize(loss_t)

        session.run(tf.initialize_all_variables())
        losses = []

        epoch = 0
        while True:
            for iter_no, (train_x, train_y) in enumerate(data.iterate_train(BATCH)):
                loss, _ = session.run([loss_t, opt_t], feed_dict={
                    ph_input: train_x,
                    ph_labels: train_y
                })
                losses.append(loss)
                if iter_no % 100 == 0:
                    log.info("Epoch=%d, iter=%d, epoch_perc=%.2f%%, mean_loss=%s",
                             epoch, iter_no, data.progress*100.0, np.array(losses).mean())
                    losses = []
            epoch += 1
    pass
