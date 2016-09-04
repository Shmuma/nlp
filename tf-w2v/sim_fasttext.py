#!/usr/bin/env python3
import argparse
import numpy as np
import logging as log


def find_similar(word, dict_data, dict_rev, model):
    word_id = dict_data.get(word.lower())
    if word_id is None:
        return None
    sim = np.dot(model, model[word_id])
    order = np.argsort(sim)
    top_idx = reversed(order[-args.top:])
    return [dict_rev[idx] for idx in top_idx if idx != word_id]


if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="FastText similarity query tool")
    parser.add_argument("-v", "--vectors", required=True, help="File name with model file to read")
    parser.add_argument("-t", "--top", type=int, default=10, help="How many similar words to show, default=10")
    parser.add_argument("-q", "--query-file", required=False, help="Query file to read input words")
    parser.add_argument("words", metavar="word", nargs="*", help="Words to find similar")
    args = parser.parse_args()

    log.info("Read vectors from %s", args.vectors)
    with open(args.vectors, "rt") as fd:
        dict_size, vec_len = map(int, fd.readline().strip().split(" "))
        dict_data = {}
        dict_rev = {}
        embeddings = np.ndarray(shape=(dict_size, vec_len), dtype=np.float32)
        for word_idx in range(dict_size):
            vals = fd.readline().strip().split(' ')
            word, vec = vals[0], list(map(float, vals[1:]))
            dict_data[word] = word_idx
            dict_rev[word_idx] = word
            embeddings[word_idx] = vec

    log.info("Read %d word vectors from model file", len(dict_data))
    log.info("Calculate normalised embeddings")
    norm_embeddings = embeddings / np.sqrt(np.sum(np.square(embeddings), axis=1, keepdims=True))

    query = []

    if args.query_file is not None:
        with open(args.query_file, "rt") as fd:
            for l in fd:
                query.append(l.strip())

    query.extend(args.words)

    for word in query:
        top = find_similar(word, dict_data, dict_rev, norm_embeddings)
        if top is None:
            log.info("Word '%s' not found in dictionary", word)
        else:
            log.info("Top%d for '%s' is: %s", args.top, word, ", ".join(top))

