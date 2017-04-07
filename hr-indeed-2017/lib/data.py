import re
import json
import smart_open
import numpy as np
import logging as log
import enum

from nltk.tokenize import TweetTokenizer


def read_runfile(name):
    with open(name, 'rt') as fd:
        return json.load(fd)


def read_embeddings(file_name):
    """
    Read embeddings from text file: http://nlp.stanford.edu/data/glove.6B.zip
    :param file_name:
    :return: tuple with (dict word->id mapping, numpy matrix with embeddings)
    """
    weights = []
    words = {'UNK': 0}
    with smart_open.smart_open(file_name, "rb") as fd:
        for idx, l in enumerate(fd):
            v = str(l, encoding='utf-8').split(' ')
            word, vec = v[0], list(map(float, v[1:]))
            words[word] = idx+1
            weights.append(vec)
    weights.insert(0, [0.0]*len(weights[0]))
    return words, np.array(weights, dtype=np.float32)


def read_samples(file_name, words):
    with smart_open.smart_open(file_name, "rb") as fd:
        # skip header
        header = str(fd.readline(), encoding='utf-8').rstrip().split('\t')
        result = {name: [] for name in header}
        for l in fd:
            entries = str(l, encoding='utf-8').rstrip().split('\t', maxsplit=len(header)-1)
            for name, val in zip(header, entries):
                result[name].append(val)

    return result


def encode_tags(tags, output):
    result = []

    for tag in tags:
        t_set = set(tag.split(' '))
        assert issubclass(output, enum.Enum)
        res_index = None
        for idx, member in enumerate(output.__members__.values()):
            if member.value == '' or member.value in t_set:
                res_index = idx
        assert res_index is not None
        v = np.zeros(len(output), dtype=np.float32)
        v[res_index] = 1.0
        result.append(v)

    return result


def tokenize_texts(texts, words):
    results = []
    for text in texts:
        t = text.lower().strip()
        t = t.replace('\n', ' ').replace('\t', ' ')
        t = t.replace("'s", " 's ")
        t = t.replace("'ll", " 'll ")
        t = t.replace('-', ' - ')
        t = t.replace('.', ' . ')
        res = TweetTokenizer(preserve_case=False, reduce_len=True).tokenize(t)
        ids = []
        for w in res:
            w_id = words.get(w)
            if w_id is None:
#                log.warning("Unknown word found: %s", w)
                w_id = 0
            ids.append(w_id)
        results.append(ids)
    return results


def iterate_batches(batch_size, texts, tags):
    ofs = 0
    while (ofs+1)*batch_size<= len(texts):
        l = ofs*batch_size
        r = (ofs+1)*batch_size
        s = texts[l:r]
        t = np.array(tags[l:r])
        yield s, t
        ofs += 1


def iterate_batch_windows(batch, window_size):
    max_len = max(map(len, batch))
    windows = (max_len + window_size-1) // window_size
    for i in range(windows):
        w = []
        for seq in batch:
            t = seq[i*window_size:(i+1)*window_size]
            t.extend([0] * (window_size - len(t)))
            w.append(t)
        yield w
