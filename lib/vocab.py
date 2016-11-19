class Vocab:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.next_idx = 0

    def build(self, tokens):
        if self.unk_token() not in self.word_to_index:
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

