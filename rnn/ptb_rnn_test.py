#!/usr/bin/env python
"""
Loads model and generates sentenses on it
"""
import argparse
import logging as log
import tensorflow as tf

import sys
sys.path.append("..")
from lib import ptb, vocab
from rnn import ptb_rnn_train as model


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="File name with model to use")
    args = parser.parse_args()

    log.info("Loading vocabulary")
    data = ptb.PTBDataset("data", vocab.Vocab(), model.NUM_STEPS)
    log.info("Loaded, creating model")

    ph_input = tf.placeholder(tf.int32, shape=(None, model.NUM_STEPS), name="input")
    initial_state, outputs, final_state = model.make_net(ph_input, data.vocab.size())
    saver = tf.train.Saver()

    with tf.Session() as session:
        log.info("Restoring the model from %s", args.model)
        saver.restore(session, args.model)
        log.info("Model restored")
    pass
