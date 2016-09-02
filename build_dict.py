"""
Tool builds dictionary and train data from given text data.
"""
import numpy as np
import argparse
import logging as log
import pickle
import struct


def iterate_input_sentences(input_file, header=False, from_col=0):
    with open(input_file, "r") as fd:
        # skip header
        if header:
            fd.readline()
        for l in fd:
            yield l.strip().lower().split()[from_col:]


def build_dict(input_file, header=False, from_col=0):
    res = {}
    next_token_id = 0
    for tokens in iterate_input_sentences(input_file, header=header, from_col=from_col):
        for token in tokens:
            if token not in res:
                res[token] = next_token_id
                next_token_id += 1
    return res


def shuffle_and_flush(buffer, fd, limit):
    np.random.shuffle(buffer[:limit])
    for ofs in range(limit):
        fd.write(struct.pack("II", buffer[ofs, 0], buffer[ofs, 1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input file with sentences (one per line with index as first token)")
    parser.add_argument("--header", action="store_true", default=False, help="Input file have header")
    parser.add_argument("--from-col", type=int, default=0, help="Start column with word data")
    parser.add_argument("-d", "--dict", help="File to save dict data", required=False)
    parser.add_argument("-t", "--train", help="File save train data (center id, context id)", required=False)
    parser.add_argument("-c", "--context", type=int, default=8, help="Context size for train data (one side), default=8")
    parser.add_argument("--shuffle-buffer", type=int, default=0, help="Shuffle buffer of examples, default=0 (disabled)")
    args = parser.parse_args()

    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Building dictionary from %s", args.input)
    dict_data = build_dict(args.input, header=args.header, from_col=args.from_col)
    log.info("Resulting dict has %d uniq tokens", len(dict_data))
    if args.dict:
        with open(args.dict, "wb+") as fd:
            pickle.dump(dict_data, fd)
        log.info("Dict saved in %s", args.dict)

    if args.train:
        buffer_ofs = 0
        if args.shuffle_buffer > 0:
            shuffle_buffer = np.ndarray(shape=(args.shuffle_buffer, 2), dtype=np.int32)
        else:
            shuffle_buffer = None

        log.info("Starting to generate training data with one-side context %d", args.context)
        train_samples = 0
        with open(args.train, "wb+") as fd:
            for sentence in iterate_input_sentences(args.input, header=args.header, from_col=args.from_col):
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
                        if shuffle_buffer is None:
                            fd.write(center_dat)
                            fd.write(struct.pack("I", context_id))
                        else:
                            shuffle_buffer[buffer_ofs, 0] = center_id
                            shuffle_buffer[buffer_ofs, 1] = context_id
                            buffer_ofs += 1
                            if buffer_ofs == args.shuffle_buffer:
                                shuffle_and_flush(shuffle_buffer, fd, buffer_ofs)
                                buffer_ofs = 0
                        train_samples += 1
            if shuffle_buffer is not None and buffer_ofs > 0:
                shuffle_and_flush(shuffle_buffer, fd, buffer_ofs)
        log.info("Generated %d train pairs", train_samples)
