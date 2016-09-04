#!/usr/bin/env python3
"""
Tool builds dictionary and train data from given text data.
"""
import numpy as np
import argparse
import logging as log
import pickle
import struct
from time import time
from datetime import timedelta


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


class TrainWriter:
    def __init__(self, file_name, shuffle_buffer_size):
        self.fd = open(file_name, "wb+")
        if shuffle_buffer_size > 0:
            self.shuffle_buffer = np.ndarray(shape=(shuffle_buffer_size, 2), dtype=np.uint32)
            self.shuffle_buffer_size = shuffle_buffer_size
            self.shuffle_buffer_ofs = 0
        else:
            self.shuffle_buffer = None

    def close(self):
        if self.shuffle_buffer is not None and self.shuffle_buffer_ofs > 0:
            self._flush()
        self.fd.close()

    def _flush(self):
        log.info("Flushing buffer with %d samples", self.shuffle_buffer_ofs)
        st_time = time()
        np.random.shuffle(self.shuffle_buffer[:self.shuffle_buffer_ofs])
        for ofs in range(self.shuffle_buffer_ofs):
            self.fd.write(struct.pack("II", *self.shuffle_buffer[ofs]))
        delta = time() - st_time
        speed = self.shuffle_buffer_ofs / delta
        log.info("Flused in %s, speed %.2f samples/sec", timedelta(seconds=delta), speed)
        self.shuffle_buffer_ofs = 0

    def append(self, center_id, context_id):
        if self.shuffle_buffer is not None:
            self.shuffle_buffer[self.shuffle_buffer_ofs, 0] = center_id
            self.shuffle_buffer[self.shuffle_buffer_ofs, 1] = context_id
            self.shuffle_buffer_ofs += 1
            if self.shuffle_buffer_ofs == self.shuffle_buffer_size:
                self._flush()
        else:
            self.fd.write(struct.pack("II", center_id, context_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input file with sentences (one per line with index as first token)")
    parser.add_argument("--header", action="store_true", default=False, help="Input file have header")
    parser.add_argument("--from-col", type=int, default=0, help="Start column with word data")
    parser.add_argument("-d", "--dict", help="File to save dict data", required=False)
    parser.add_argument("-t", "--train", help="File save train data (center id, context id)", required=False)
    parser.add_argument("-c", "--context", type=int, default=8, help="Context size for train data (one side), default=8")
    parser.add_argument("--skips-per-window", type=int, default=None, help="Count of skipgrams generated from the window, default=All")
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
        train_writer = TrainWriter(args.train, args.shuffle_buffer)
    else:
        train_writer = None

    train_samples = 0
    log.info("Starting to generate training data with one-side context %d", args.context)

    for sentence in iterate_input_sentences(args.input, header=args.header, from_col=args.from_col):
        s_len = len(sentence)
        # treat every word in a sentence as a center
        for center_ofs, center_word in enumerate(sentence):
            left = max(0, center_ofs - args.context)
            right = min(s_len, center_ofs + args.context)
            center_id = dict_data[center_word]
            # iterate for context words
            context_ids = list(filter(lambda idx: idx != center_id,
                                      map(lambda w: dict_data[w], sentence[left:right+1])))
            np.random.shuffle(context_ids)
            if args.skips_per_window is not None:
                context_ids = context_ids[:args.skips_per_window]
            for context_id in context_ids:
                train_writer.append(center_id, context_id)
                train_samples += 1

    if train_writer:
        train_writer.close()
        train_writer = None
    log.info("Generated %d train pairs", train_samples)
