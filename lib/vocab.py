class Vocab:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.next_idx = 0

    def build(self, tokens):
        self._add_word(self.unk_token())
        for token in tokens:
            if token in self.word_to_index:
                continue
            self._add_word(token)

    def _add_word(self, token):
        self.word_to_index[token] = self.next_idx
        self.index_to_word[self.next_idx] = token
        self.next_idx += 1

    def size(self):
        return len(self.word_to_index)

    def __str__(self):
        return "Vocab[size=%d]" % self.size()

    def unk_token(self):
        return '<unk>'

    def eos_token(self):
        return '<eos>'

    def encode(self, token):
        res = self.word_to_index.get(token)
        if res is None:
            return self.encode(self.unk_token())
        return res

    def decode(self, token_id):
        return self.index_to_word.get(token_id, self.unk_token())


class FastTextVocab:
    def __init__(self, file_name):
        self.word_to_index = {}
        self.index_to_word = {}
        self._read_fasttext(file_name)

    def build(self, tokens):
        for token in tokens:
            assert token in self.word_to_index

    def _read_fasttext(self, file_name):
        with open(file_name, "rt", encoding='utf-8') as fd:
            dims = list(map(int, fd.readline().split()))
            for word_id in range(dims[0]):
                word = fd.readline().split()[0]
                self.word_to_index[word] = word_id
                self.index_to_word[word_id] = word

    def size(self):
        return len(self.word_to_index)

    def __str__(self):
        return "FastTextVocab[size=%d]" % self.size()

    def unk_token(self):
        return '<unk>'

    def eos_token(self):
        return '</s>'

    def encode(self, token):
        res = self.word_to_index.get(token)
        if res is None:
            return self.encode(self.unk_token())
        return res

    def decode(self, token_id):
        return self.index_to_word.get(token_id, self.unk_token())
