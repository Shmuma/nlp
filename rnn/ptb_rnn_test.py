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



def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))



def test_sentence(vocab, session, ph_input, initial_state, outputs, final_state, sentence, ph_dropout, max_steps=100):
    tokens = [vocab.encode(word) for word in sentence.lower().split()]
    prob = None
    state = initial_state.eval()
    prob_t = tf.nn.softmax(tf.cast(outputs[0], 'float64'))
    # feed whole sentence to get state
    for t in tokens:
        state, prob = session.run([final_state, prob_t], feed_dict={
            ph_input: [[t]],
            ph_dropout: 1.0,
            initial_state: state
        })

    res_tokens = sentence.lower().split()
    for _ in range(max_steps):
        new_token_id = sample(prob[0])
        new_token = vocab.decode(new_token_id)
        if new_token == vocab.eos_token():
            break
        res_tokens.append(new_token)
        feed_dict = {
            ph_input: [[new_token_id]],
            initial_state: state,
            ph_dropout: 1.0
        }
        state, prob = session.run([final_state, prob_t], feed_dict=feed_dict)
    print(" ".join(res_tokens))


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="File name with model to use")
    parser.add_argument("-s", "--sentence", help="Sentence to feed to model and continue")
    parser.add_argument("-n", "--number", type=int, default=5, help="Count of sentence generations")
    args = parser.parse_args()

    log.info("Loading vocabulary")
    data = ptb.PTBDataset("data", vocab.Vocab(), batch_size=1)
    log.info("Loaded vocab %s, creating model", data.vocab)

    ph_input, initial_state, outputs, final_state, ph_dropout = model.make_net(data.vocab.size(), batch=1, num_steps=1)
    saver = tf.train.Saver()

    with tf.Session() as session:
        log.info("Restoring the model from %s", args.model)
        saver.restore(session, args.model)
        log.info("Model restored")

        if args.sentence:
            for _ in range(args.number):
                test_sentence(data.vocab, session, ph_input, initial_state, outputs, final_state, args.sentence, ph_dropout)
    pass
