import os
import random
import numpy as np


class PTBDataset:
    def __init__(self, data_dir, vocab, num_steps):
        self.num_steps = num_steps
        self.data_dir = data_dir

        self.vocab = vocab
        vocab.build(self._read_train())

        self.train_x, self.train_y = self._build_samples(self._read_train())
        self.progress = 0.0

    def __str__(self):
        return "PTBData[vocab=%s, train_samples=%d]" % (self.vocab, len(self.train_x))

    def _read_train(self):
        with open(os.path.join(self.data_dir, "ptb.train.txt"), "rt", encoding='utf-8') as fd:
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
        """
        Iterate for train samples batched with given batch size
        """
        if shuffle:
            shuffle_idx = list(range(len(self.train_x)))
            random.shuffle(shuffle_idx)
            self.train_x = self.train_x[shuffle_idx]
            self.train_y = self.train_y[shuffle_idx]

        ofs = 0
        self.progress = 0.0
        while ofs + batch_size <= len(self.train_x):
            x = self.train_x[ofs:ofs+batch_size]
            y = self.train_y[ofs:ofs+batch_size]
            yield x, y
            ofs += batch_size
            self.progress = ofs / len(self.train_x)
