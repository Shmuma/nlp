#!/usr/bin/env python
import os
import time
import datetime
import json
import argparse
import logging as log

import numpy as np
from keras.optimizers import Adam, RMSprop, Adagrad, Nadam
import keras.backend as K
import tensorflow as tf

from lib import data
from lib import model
from lib import utils

INPUT_TRAIN = "data/train.tsv.gz"
BATCH_SIZE = 256
TEST_BATCH_SIZE = 16


def split_test_train(full_tokens, full_tags, ratio, batch_size):
    """
    Perform split of input data into train/test set. Make sure that train dataset is rounded by batch_size
    :param full_tokens:
    :param full_tags:
    :param ratio:
    :param batch_size:
    :return:
    """
    full_size = len(full_tokens)
    train_size = int((full_size * (1.0 - ratio) // batch_size) * batch_size)
    train_tokens, test_tokens = full_tokens[:train_size], full_tokens[train_size:]
    train_tags, test_tags = full_tags[:train_size], full_tags[train_size:]
    return train_tokens, train_tags, test_tokens, test_tags


def sort_examples(tokens, tags):
    l = list(zip(tokens, tags))
    l.sort(key=lambda p: len(p[0]))
    return zip(*l)

# Batch=1
# 2017-04-04 06:44:21,800 INFO Test done in 0:01:23.937751, F1 score = 0.19446
# Batch=16
# 2017-04-04 06:55:55,528 INFO Test done in 0:00:12.981350, F1 score = 0.06667
def test_accuracy(mod, test_tokens, test_tags, embeddings):
    stp = sfp = sfn = 0

    for batch_seq, batch_tags in data.iterate_batches(TEST_BATCH_SIZE, test_tokens, test_tags):
        last_pred = None
        for win in data.iterate_batch_windows(batch_seq, model.WINDOW_SIZE):
            input_x = [embeddings[x] for x in win]
            last_pred = mod.predict_on_batch(np.array(input_x))
        mod.reset_states()
        for pred_arr, true_arr in zip(last_pred, batch_tags):
            stp, sfp, sfn = model.tags_compare(pred_arr, true_arr, stp, sfp, sfn)

    if stp == 0 or stp + sfp == 0 or stp + sfn == 0:
        return 0.0
    p = stp / (stp + sfp)
    r = stp / (stp + sfn)

    return 2.0 * p * r / (p+r)


if __name__ == "__main__":
    log.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=log.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runfile", required=True, help="Name of runfile to use")
    parser.add_argument("-n", "--name", required=True, help="Name of run, used to save logs and models")
    parser.add_argument("-t", "--type", choices=sorted(model.OUTPUTS.keys()), required=True,
                        help="Name of net we're going to train")
    args = parser.parse_args()
    output = model.OUTPUTS[args.type]

    runfile = data.read_runfile(args.runfile)
    log.info("Runfile: %s", json.dumps(runfile, indent='\t', sort_keys=True))

    log.info("Reading embeddings, it can take time")
    words, embeddings = data.read_embeddings(runfile['embeddings'])
    emb_len = embeddings.shape[1]
    log.info("We've read %d words from embeddings, it's size = %d", len(words), emb_len)

    log.info("Reading input samples and preprocess it")
    train_data = data.read_samples(INPUT_TRAIN, words)
    # split data on train/test sets
    train_tokens, train_tags, test_tokens, test_tags = split_test_train(train_data['description'], train_data['tags'],
                                                                        0.1, BATCH_SIZE)
    log.info("Train has %d samples, test %d samples", len(train_tokens), len(test_tokens))

    train_tokens = data.tokenize_texts(train_tokens, words)
    # sort train samples by tokens count
    train_tokens, train_tags = sort_examples(train_tokens, train_tags)

    train_tags = data.encode_tags(train_tags, output)
    test_tokens = data.tokenize_texts(test_tokens, words)
    test_tags = data.encode_tags(test_tags, output)

    log.info("Create model")
    mod = model.create_model(BATCH_SIZE, emb_len, output)
    optimiser = Adam(lr=0.001, decay=0.00001)  # 0.00003 = approx 0.99 per epoch
    mod.compile(optimizer=optimiser, loss='categorical_crossentropy')
    mod.summary()

    test_mod = model.create_model(TEST_BATCH_SIZE, emb_len, output)

    summary_writer = tf.summary.FileWriter("logs/" + args.name)
#    utils.summarize_gradients(mod)
#      mod.metrics_names.append("summary")
#      mod.metrics_tensors.append(tf.summary.merge_all())

    # iterate sequences
    best_f1 = 0
    for epoch_idx in range(runfile['epoches']):
        lr = K.get_value(optimiser.lr) * (1.0 / (1.0 + optimiser.initial_decay * K.get_value(optimiser.iterations)))
        log.info("Epoch %d, iter=%d, lr=%f", epoch_idx, K.get_value(optimiser.iterations), lr)
        l_agg = {}
        summ = None

        for batch_seq, batch_tags in data.iterate_batches(BATCH_SIZE, train_tokens, train_tags):
            # process all sequence windows
            for seq_window in data.iterate_batch_windows(batch_seq, model.WINDOW_SIZE):
                input_x = [embeddings[x] for x in seq_window]
                loss = mod.train_on_batch(np.array(input_x), batch_tags)
                l_dict = dict(zip(mod.metrics_names, [loss]))
                for l_name, l_val in l_dict.items():
                    if l_name == 'summary':
                        summ = l_val
                    else:
                        if l_name not in l_agg:
                            l_agg[l_name] = []
                        l_agg[l_name].append(l_val)

            mod.reset_states()

        # save summary
        for n, v in l_agg.items():
            utils.summary_value(n + "_mean", np.mean(v), summary_writer, epoch_idx)
            utils.summary_value(n + "_max", np.max(v), summary_writer, epoch_idx)
            utils.summary_value(n + "_min", np.min(v), summary_writer, epoch_idx)
        summary_writer.add_summary(summ, global_step=epoch_idx)
        summary_writer.flush()

        # do test
        test_mod.set_weights(mod.get_weights())
        test_start = time.time()
        f1 = test_accuracy(test_mod, test_tokens, test_tags, embeddings)
        test_d = datetime.timedelta(seconds=time.time() - test_start)

        log.info("Test done in %s, F1 score = %.5f", test_d, f1)
        utils.summary_value("f1", f1, summary_writer, epoch_idx)
        utils.summary_value("lr", lr, summary_writer, epoch_idx)
        summary_writer.flush()
        if f1 > best_f1:
            if f1 > 0.5:
                mod.save_weights(os.path.join("logs", args.name, "model-f1=%.4f.m5" % f1))
                mod.save_weights(os.path.join("logs", args.name, "model-best.m5"))
            best_f1 = f1
    pass
