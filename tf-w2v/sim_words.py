#!/usr/bin/env python3
import argparse
import numpy as np
import logging as log
import pickle


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="File name with model file to read")
    parser.add_argument("-d", "--dict", required=True, help="Dictionary file")
    parser.add_argument("-t", "--top", type=int, default=10, help="How many similar words to show")
    parser.add_argument("words", metavar="word", nargs="+", help="Words to find similar")
    args = parser.parse_args()

    log.info("Reading model from %s", args.model)
    model = np.load(args.model)
    log.info("Reading dict from %s", args.dict)
    with open(args.dict, "rb") as fd:
        dict_data = pickle.load(fd)
    log.info("Building reverse dictionary")
    dict_rev = {idx: token for token, idx in dict_data.items()}

    log.info("Model shape: %s, dict size: %d", model.shape, len(dict_data))

    for word in args.words:
        word_id = dict_data.get(word.lower())
        if word_id is None:
            log.info("Word '%s' not found in dictionary", word)
            continue
        sim = np.dot(model, model[word_id])
        order = np.argsort(sim)
        top_idx = reversed(order[-args.top-1:])
        top = [dict_rev[idx] for idx in list(top_idx)[1:]]
        log.info("Top%d for '%s' is: %s", args.top, word, ", ".join(top))
