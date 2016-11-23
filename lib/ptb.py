import os
import random
import numpy as np


class PTBDataset:
    TRAIN_FILE = "ptb.train.txt"
    VALID_FILE = "ptb.valid.txt"
    TEST_FILE =  "ptb.test.txt"

    def __init__(self, data_dir, vocab, num_steps):
        self.num_steps = num_steps
        self.data_dir = data_dir

        self.vocab = vocab
        vocab.build(self._read_file(self.TRAIN_FILE))

        self.train_x = self.train_y = None
        self.valid_x = self.valid_y = None
        self.test_x = self.test_y = None

    def __str__(self):
        return "PTBData[vocab=%s, train_samples=%d]" % (self.vocab, len(self.train_x))

    def load_dataset(self):
        self.train_x, self.train_y = self._build_samples(self._read_file(self.TRAIN_FILE))
        self.valid_x, self.valid_y = self._build_samples(self._read_file(self.VALID_FILE))
        self.test_x, self.test_y = self._build_samples(self._read_file(self.TEST_FILE))

    def _read_file(self, file_name):
        with open(os.path.join(self.data_dir, file_name), "rt", encoding='utf-8') as fd:
            for l in fd:
                l = l.strip()
                for w in l.split():
                    yield w
                yield self.vocab.eos_token()

    def _build_samples(self, tokens_iter, limit=None):
        res_x = []
        res_y = []
        window = []
        for token in tokens_iter:
            token_id = self.vocab.encode(token)

            # push new entry
            prev_window = list(window)
            window.append(token_id)
            if len(window) > self.num_steps:
                window = window[-self.num_steps:]

            # if we have complete data sample, append
            if len(prev_window) == self.num_steps:
                res_x.append(np.array(prev_window))
                res_y.append(np.array(window))
                if limit is not None and len(res_x) == limit:
                    break

            # cleanup window at the end of sentence
            if token == self.vocab.eos_token():
                window = []
        return np.array(res_x), np.array(res_y)

    def iterate_train(self, batch_size, shuffle=True):
        for v in self.iterate_dataset(batch_size, self.train_x, self.train_y, shuffle):
            yield v

    def iterate_validation(self, batch_size):
        for v in self.iterate_dataset(batch_size, self.valid_x, self.valid_y, shuffle=False):
            yield v

    def iterate_test(self, batch_size):
        for v in self.iterate_dataset(batch_size, self.test_x, self.test_y, shuffle=False):
            yield v

    def iterate_dataset(self, batch_size, x, y, shuffle=True):
        """
        Iterate for train samples batched with given batch size
        """
        if shuffle:
            shuffle_idx = list(range(len(x)))
            random.shuffle(shuffle_idx)
            x = x[shuffle_idx]
            y = y[shuffle_idx]

        ofs = 0
        progress = 0.0
        while ofs + batch_size <= len(x):
            xx = x[ofs:ofs+batch_size]
            yy = y[ofs:ofs+batch_size]
            yield xx, yy, progress
            ofs += batch_size
            progress = ofs / len(x)


