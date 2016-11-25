#!/usr/bin/env python
"""
Loads model and generates sentenses on it
"""
import argparse
import numpy as np
import logging as log
import tensorflow as tf

import sys
sys.path.append("..")
from lib import ptb, vocab
from rnn import ptb_rnn_train as model


def test_sentence(vocab, session, ph_input, initial_state, outputs, final_state, sentence, ph_dropout, max_steps=100):
    tokens = [vocab.encode(word) for word in sentence.lower().split()]
    pred = None
    pred_t = tf.nn.softmax(outputs[0])
    state = initial_state.eval()
    # feed whole sentence to get state
    for t in tokens:
        state, pred = session.run([final_state, pred_t], feed_dict={
            ph_input: [[t]],
            ph_dropout: 1.0,
            initial_state: state
        })

    res_tokens = []
    for _ in range(max_steps):
        new_token_id = np.argmax(pred)
        new_token = vocab.decode(new_token_id)
#        if new_token == vocab.eos_token():
#            break
        res_tokens.append(new_token)
        feed_dict = {
            ph_input: [[new_token_id]],
            initial_state: state,
            ph_dropout: 1.0
        }
        state, pred = session.run([final_state, pred_t], feed_dict=feed_dict)
    print(" ".join(res_tokens))


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="File name with model to use")
    parser.add_argument("-s", "--sentence", help="Sentence to feed to model and continue")
    args = parser.parse_args()

    log.info("Loading vocabulary")
    data = ptb.PTBDataset("data", vocab.Vocab(), batch_size=1)
    log.info("Loaded, creating model")

    ph_input, initial_state, outputs, final_state, ph_dropout = model.make_net(data.vocab.size(), num_steps=1, batch=1)
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        log.info("Restoring the model from %s", args.model)
        saver.restore(session, args.model)
        log.info("Model restored")

        if args.sentence:
            test_sentence(data.vocab, session, ph_input, initial_state, outputs, final_state, args.sentence, ph_dropout)
    pass
