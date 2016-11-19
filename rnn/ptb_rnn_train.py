import tensorflow as tf
import logging as log

import sys
sys.path.append("..")

from lib import ptb, vocab

BATCH = 64
NUM_STEPS = 10
EMBEDDING = 100
CELL_SIZE = 100

if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Loading PTB dataset...")
    data = ptb.PTBDataset("data", vocab.Vocab(), num_steps=10)
    log.info("Loaded: %s", data)

    with tf.Session() as session:
        ph_input = tf.placeholder(tf.int32, shape=(None, NUM_STEPS), name="input")
        cell = tf.nn.rnn_cell.BasicRNNCell(CELL_SIZE)
        initial_state = cell.zero_state(BATCH, dtype=tf.float32)

        with tf.variable_scope("Model"):
            embedding = tf.get_variable("embedding", [data.vocab.size(), EMBEDDING])
            inputs = tf.nn.embedding_lookup(embedding, ph_input)
            inputs = [tf.squeeze(val, squeeze_dims=[1]) for val in tf.split(split_dim=1, num_split=NUM_STEPS, value=inputs)]
        print(inputs)
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=initial_state)
        print(outputs)
        print(state)

#        for train_x, train_y in data.iterate_train(BATCH):
#            print(len(train_x), len(train_y))
    pass
