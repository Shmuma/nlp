import os
import numpy as np


class PTBDataset:
    TRAIN_FILE = "ptb.train.txt"
    VALID_FILE = "ptb.valid.txt"
    TEST_FILE =  "ptb.test.txt"

    def __init__(self, data_dir, vocab, batch_size):
        self.batch_steps = batch_size
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

    def _build_samples(self, tokens_iter):
        data = [self.vocab.encode(token) for token in tokens_iter]
        batch_item_len = len(data) // self.batch_steps
        even_len = self.batch_steps * batch_item_len
        x = np.array(data[:even_len])
        y = np.array(data[1:even_len+1])
        x = np.reshape(x, (self.batch_steps, batch_item_len))
        y = np.reshape(y, (self.batch_steps, batch_item_len))
        return x, y

    def iterate_train(self, num_steps):
        for v in self.iterate_dataset(num_steps, self.train_x, self.train_y):
            yield v

    def iterate_validation(self, num_steps):
        for v in self.iterate_dataset(num_steps, self.valid_x, self.valid_y):
            yield v

    def iterate_test(self, num_steps):
        for v in self.iterate_dataset(num_steps, self.test_x, self.test_y):
            yield v

    def iterate_dataset(self, num_steps, x, y):
        """
        Iterate for train samples batched with given batch size
        """
        ofs = 0
        max_width = x.shape[1]
        progress = 0.0
        while ofs + num_steps <= max_width:
            xx = x[:, ofs:ofs+num_steps]
            yy = y[:, ofs:ofs+num_steps]
            yield xx, yy, progress
            ofs += num_steps
            progress = ofs / max_width


