"""
Tool builds dictionary and train data from given text data.
"""

import argparse
import logging as log
import pickle
import struct


def iterate_input_sentences(input_file):
    with open(input_file, "r") as fd:
        # skip header
        fd.readline()
        for l in fd:
            yield l.strip().lower().split()[1:]


def build_dict(input_file):
    res = {}
    next_token_id = 0
    for tokens in iterate_input_sentences(input_file):
        for token in tokens:
            if token not in res:
                res[token] = next_token_id
                next_token_id += 1
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input file with sentences (one per line with index as first token)")
    parser.add_argument("-d", "--dict", help="File to save dict data", required=False)
    parser.add_argument("-t", "--train", help="File save train data (center id, context id)", required=False)
    parser.add_argument("-c", "--context", type=int, default=2, help="Context size for train data (one side), default=2")
    args = parser.parse_args()

    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Building dictionary from %s", args.input)
    dict_data = build_dict(args.input)
    log.info("Resulting dict has %d uniq tokens", len(dict_data))
    if args.dict:
        with open(args.dict, "wb+") as fd:
            pickle.dump(dict_data, fd)
        log.info("Dict saved in %s", args.dict)

    if args.train:
        log.info("Starting to generate training data with one-side context %d", args.context)
        train_samples = 0
        with open(args.train, "wb+") as fd:
            for sentence in iterate_input_sentences(args.input):
                s_len = len(sentence)
                # treat every word in a sentence as a center
                for center_ofs, center_word in enumerate(sentence):
                    left = max(0, center_ofs - args.context)
                    right = min(s_len, center_ofs + args.context)
                    center_id = dict_data[center_word]
                    center_dat = struct.pack("I", center_id)
                    # iterate for context words
                    for context_word in sentence[left:right+1]:
                        context_id = dict_data[context_word]
                        if context_id == center_id:
                            continue
                        context_dat = struct.pack("I", context_id)
                        fd.write(center_dat)
                        fd.write(context_dat)
                        train_samples += 1
        log.info("Generated %d train pairs", train_samples)