import tensorflow as tf
import numpy as np


class EmbeddingFastTextWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self, cell, embeddings_file):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self._embeddings = self._read_embeddings(embeddings_file)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the cell on embedded inputs."""
        with tf.variable_scope(scope or type(self).__name__):
            with tf.device("/cpu:0"):
                embedded = tf.nn.embedding_lookup(self._embeddings, tf.reshape(inputs, [-1]))
        return self._cell(embedded, state)

    def _read_embeddings(self, file_name):
        with open(file_name, "rt", encoding='utf-8') as fd:
            dims = list(map(int, fd.readline().split()))
            assert len(dims) == 2
            result = np.ndarray(shape=dims, dtype=np.float32)
            for token_id in range(dims[0]):
                l = fd.readline()
                vals = list(map(float, l.split()[1:]))
                assert len(vals) == dims[1]
                result[token_id] = vals
        return result
