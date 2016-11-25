#!/usr/bin/env python
"""
Trains simple language model using RNN and PTB dataset
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.seq2seq import sequence_loss
import logging as log

import sys
sys.path.append("..")

from lib import ptb, vocab

from lib.utils import calculate_perplexity, get_ptb_dataset, Vocab
from lib.utils import ptb_iterator, sample


BATCH = 64
NUM_STEPS = 10
EMBEDDING = 50
CELL_SIZE = 100
LR = 0.001
DROPOUT = 0.9

LOG_DIR = "logs"
SAVE_DIR = "saves"


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    batch_size = 64
    embed_size = 50
    hidden_size = 100
    num_steps = 10
    max_epochs = 16
    early_stopping = 2
    dropout = 0.9
    lr = 0.001


class RNNLM_Model:
    def load_data(self, debug=False):
        """Loads starter word-vectors and train/dev/test data."""
        self.vocab = Vocab()
        self.vocab.construct(get_ptb_dataset('train'))
        self.encoded_train = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('train')],
            dtype=np.int32)
        self.encoded_valid = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
            dtype=np.int32)
        self.encoded_test = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('test')],
            dtype=np.int32)
        if debug:
            num_debug = 1024
            self.encoded_train = self.encoded_train[:num_debug]
            self.encoded_valid = self.encoded_valid[:num_debug]
            self.encoded_test = self.encoded_test[:num_debug]

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building
        code and will be fed data during training.  Note that when "None" is in a
        placeholder's shape, it's flexible

        Adds following nodes to the computational graph.
        (When None is in a placeholder's shape, it's flexible)

        input_placeholder: Input placeholder tensor of shape
                           (None, num_steps), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape
                            (None, num_steps), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar),
                             type tf.float32

        Add these placeholders to self as the instance variables

          self.input_placeholder
          self.labels_placeholder
          self.dropout_placeholder

        (Don't change the variable names)
        """
        ### YOUR CODE HERE
        self.input_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps], name='Input')
        self.labels_placeholder = tf.placeholder(
            tf.int32, shape=[None, self.config.num_steps], name='Target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        ### END YOUR CODE

    def add_embedding(self):
        """Add embedding layer.

        Hint: This layer should use the input_placeholder to index into the
              embedding.
        Hint: You might find tf.nn.embedding_lookup useful.
        Hint: You might find tf.split, tf.squeeze useful in constructing tensor inputs
        Hint: Check the last slide from the TensorFlow lecture.
        Hint: Here are the dimensions of the variables you will need to create:

          L: (len(self.vocab), embed_size)

        Returns:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        """
        # The embedding lookup is currently only implemented for the CPU
        with tf.device('/cpu:0'):
            ### YOUR CODE HERE
            embedding = tf.get_variable(
                'Embedding',
                [len(self.vocab), self.config.embed_size], trainable=True)
            inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            inputs = [
                tf.squeeze(x, [1]) for x in tf.split(1, self.config.num_steps, inputs)]
            ### END YOUR CODE
            return inputs

    def add_projection(self, rnn_outputs):
        """Adds a projection layer.

        The projection layer transforms the hidden representation to a distribution
        over the vocabulary.

        Hint: Here are the dimensions of the variables you will need to
              create

              U:   (hidden_size, len(vocab))
              b_2: (len(vocab),)

        Args:
          rnn_outputs: List of length num_steps, each of whose elements should be
                       a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each a tensor of shape
                   (batch_size, len(vocab)
        """
        ### YOUR CODE HERE
        with tf.variable_scope('Projection'):
            U = tf.get_variable(
                'Matrix', [self.config.hidden_size, len(self.vocab)])
            proj_b = tf.get_variable('Bias', [len(self.vocab)])
            outputs = [tf.matmul(o, U) + proj_b for o in rnn_outputs]
        ### END YOUR CODE
        return outputs

    def add_loss_op(self, output):
        """Adds loss ops to the computational graph.

        Hint: Use tensorflow.python.ops.seq2seq.sequence_loss to implement sequence loss.

        Args:
          output: A tensor of shape (None, self.vocab)
        Returns:
          loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE
        all_ones = [tf.ones([self.config.batch_size * self.config.num_steps])]
        log.info("Loss info:")
        log.info("Output: %s", output)
        labels = tf.reshape(self.labels_placeholder, [-1])
        log.info("Labels: %s", labels)

# 2016-11-25 12:19:29,690 INFO Loss info:
# 2016-11-25 12:19:29,690 INFO Output: Tensor("RNNLM/Reshape:0", shape=(640, 10000), dtype=float32)
# 2016-11-25 12:19:29,691 INFO Labels: Tensor("RNNLM/Reshape_1:0", shape=(?,), dtype=int32)

        cross_entropy = sequence_loss(
            [output], [labels], all_ones)
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.AdamOptimizer for this model.
              Calling optimizer.minimize() will return a train_op object.

        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        ### YOUR CODE HERE
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(self.calculate_loss)
        ### END YOUR CODE
        return train_op

    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
#        self.inputs = self.add_embedding()
        self.outputs = self.add_model()
#        self.outputs = self.add_projection(self.rnn_outputs)

        # We want to check how well we correctly predict the next word
        # We cast o to float64 as there are numerical issues at hand
        # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
#        self.predictions = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs]
        # Reshape the output into len(vocab) sized chunks - the -1 says as many as
        # needed to evenly divide
        output = tf.reshape(tf.concat(1, self.outputs), [-1, len(self.vocab)])
        self.calculate_loss = self.add_loss_op(output)
        self.train_step = self.add_training_op(self.calculate_loss)
        self.global_step = 0

    def add_model(self):
        """Creates the RNN LM model.

        In the space provided below, you need to implement the equations for the
        RNN LM model. Note that you may NOT use built in rnn_cell functions from
        tensorflow.

        Hint: Use a zeros tensor of shape (batch_size, hidden_size) as
              initial state for the RNN. Add this to self as instance variable

              self.initial_state

              (Don't change variable name)
        Hint: Add the last RNN output to self as instance variable

              self.final_state

              (Don't change variable name)
        Hint: Make sure to apply dropout to the inputs and the outputs.
        Hint: Use a variable scope (e.g. "RNN") to define RNN variables.
        Hint: Perform an explicit for-loop over inputs. You can use
              scope.reuse_variables() to ensure that the weights used at each
              iteration (each time-step) are the same. (Make sure you don't call
              this for iteration 0 though or nothing will be initialized!)
        Hint: Here are the dimensions of the various variables you will need to
              create:

              H: (hidden_size, hidden_size)
              I: (embed_size, hidden_size)
              b_1: (hidden_size,)

        Args:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size)
        """
        ### YOUR CODE HERE
        # with tf.variable_scope('InputDropout'):
        #     inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]

        cell = tf.nn.rnn_cell.BasicRNNCell(self.config.hidden_size, activation=tf.sigmoid)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.dropout_placeholder,
                                             output_keep_prob=self.dropout_placeholder)
        cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, len(self.vocab), self.config.embed_size,
                                               initializer=tf.contrib.layers.xavier_initializer())
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, len(self.vocab))
        self.initial_state = cell.zero_state(self.config.batch_size, dtype=tf.float32)
        inputs = [tf.squeeze(val, squeeze_dims=[1]) for val in
                  tf.split(split_dim=1, num_split=self.config.num_steps, value=self.input_placeholder)]
        rnn_outputs, self.final_state = tf.nn.rnn(cell, inputs, initial_state=self.initial_state)

        # with tf.variable_scope('RNN') as scope:
        #     self.initial_state = tf.zeros(
        #         [self.config.batch_size, self.config.hidden_size])
        #     state = self.initial_state
        #     rnn_outputs = []
        #     for tstep, current_input in enumerate(inputs):
        #         if tstep > 0:
        #             scope.reuse_variables()
        #         RNN_H = tf.get_variable(
        #             'HMatrix', [self.config.hidden_size, self.config.hidden_size])
        #         RNN_I = tf.get_variable(
        #             'IMatrix', [self.config.embed_size, self.config.hidden_size])
        #         RNN_b = tf.get_variable(
        #             'B', [self.config.hidden_size])
        #         state = tf.nn.sigmoid(
        #             tf.matmul(state, RNN_H) + tf.matmul(current_input, RNN_I) + RNN_b)
        #         rnn_outputs.append(state)
        #     self.final_state = rnn_outputs[-1]

        # with tf.variable_scope('RNNDropout'):
        #     rnn_outputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in rnn_outputs]
        ### END YOUR CODE
        return rnn_outputs

    def run_epoch(self, session, data, train_op=None, verbose=100, summary_hook=None):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in ptb_iterator(data, config.batch_size, config.num_steps))
        total_loss = []
        state = self.initial_state.eval()
        for step, (x, y) in enumerate(
                ptb_iterator(data, config.batch_size, config.num_steps)):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history
            feed = {self.input_placeholder: x,
                    self.labels_placeholder: y,
                    self.initial_state: state,
                    self.dropout_placeholder: dp}
            loss, state, _ = session.run(
                [self.calculate_loss, self.final_state, train_op], feed_dict=feed)
            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
                if summary_hook is not None:
                    summary_hook(self.global_step, np.exp(np.mean(total_loss)))
            if summary_hook:
                self.global_step += 1
        if verbose:
            sys.stdout.write('\r')
        return np.exp(np.mean(total_loss))


def make_net(vocab_size, dropout_prob=DROPOUT, num_steps=NUM_STEPS, batch=BATCH):
    ph_input = tf.placeholder(tf.int32, shape=(None, NUM_STEPS), name="input")

    with tf.variable_scope("Net", initializer=None):
        cell = tf.nn.rnn_cell.BasicRNNCell(CELL_SIZE, activation=tf.sigmoid)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout_prob, output_keep_prob=dropout_prob)
        cell = tf.nn.rnn_cell.EmbeddingWrapper(cell, vocab_size, EMBEDDING,
                                               initializer=tf.contrib.layers.xavier_initializer())
        cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, vocab_size)
        initial_state = cell.zero_state(batch, dtype=tf.float32)

        inputs = [tf.squeeze(val, squeeze_dims=[1]) for val in tf.split(split_dim=1, num_split=num_steps, value=ph_input)]
        outputs, state = tf.nn.rnn(cell, inputs, initial_state=initial_state)
    return ph_input, initial_state, outputs, state


# def make_net(vocab_size, dropout_prob=DROPOUT):
#     ph_input = tf.placeholder(tf.int32, shape=(None, NUM_STEPS), name="input")
#
#     # embeddings
#     with tf.device('/cpu:0'):
#         embedding = tf.get_variable(
#             'Embedding',
#             [vocab_size, EMBEDDING], trainable=True)
#         inputs = tf.nn.embedding_lookup(embedding, ph_input)
#         inputs = [tf.squeeze(x, [1]) for x in tf.split(1, NUM_STEPS, inputs)]
#
#     with tf.variable_scope('InputDropout'):
#         inputs = [tf.nn.dropout(x, dropout_prob) for x in inputs]
#
#     with tf.variable_scope('RNN') as scope:
#         initial_state = tf.zeros([BATCH, CELL_SIZE])
#         state = initial_state
#         rnn_outputs = []
#         for tstep, current_input in enumerate(inputs):
#             if tstep > 0:
#                 scope.reuse_variables()
#             RNN_H = tf.get_variable(
#                 'HMatrix', [CELL_SIZE, CELL_SIZE])
#             RNN_I = tf.get_variable(
#                 'IMatrix', [EMBEDDING, CELL_SIZE])
#             RNN_b = tf.get_variable(
#                 'B', [CELL_SIZE])
#             state = tf.nn.sigmoid(
#                 tf.matmul(state, RNN_H) + tf.matmul(current_input, RNN_I) + RNN_b)
#             rnn_outputs.append(state)
#         final_state = rnn_outputs[-1]
#
#     with tf.variable_scope('RNNDropout'):
#         rnn_outputs = [tf.nn.dropout(x, dropout_prob) for x in rnn_outputs]
#
#     with tf.variable_scope('Projection'):
#         U = tf.get_variable(
#             'Matrix', [CELL_SIZE, vocab_size])
#         proj_b = tf.get_variable('Bias', [vocab_size])
#         outputs = [tf.matmul(o, U) + proj_b for o in rnn_outputs]
#
#     return ph_input, initial_state, outputs, final_state

class Data:
    def __init__(self):
        self.vocab = Vocab()
        self.vocab.construct(get_ptb_dataset('train'))
        self.encoded_train = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('train')],
            dtype=np.int32)
        self.encoded_valid = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('valid')],
            dtype=np.int32)
        self.encoded_test = np.array(
            [self.vocab.encode(word) for word in get_ptb_dataset('test')],
            dtype=np.int32)


def show_variables():
    log.info("Trainable variables:")
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        log.info("  %s: %s", var.name, var.get_shape())


"""
CS224N implementation:
2016-11-25 11:13:49,394 INFO Trainable variables:
2016-11-25 11:13:49,395 INFO   RNNLM/Embedding:0: (10000, 50)
2016-11-25 11:13:49,395 INFO   RNNLM/RNN/HMatrix:0: (100, 100)
2016-11-25 11:13:49,395 INFO   RNNLM/RNN/IMatrix:0: (50, 100)
2016-11-25 11:13:49,395 INFO   RNNLM/RNN/B:0: (100,)
2016-11-25 11:13:49,395 INFO   RNNLM/Projection/Matrix:0: (100, 10000)
2016-11-25 11:13:49,395 INFO   RNNLM/Projection/Bias:0: (10000,)

TF implementation:
2016-11-25 11:15:48,107 INFO Trainable variables:
2016-11-25 11:15:48,107 INFO   Net/RNN/EmbeddingWrapper/embedding:0: (10000, 50)
2016-11-25 11:15:48,107 INFO   Net/RNN/BasicRNNCell/Linear/Matrix:0: (150, 100)
2016-11-25 11:15:48,107 INFO   Net/RNN/BasicRNNCell/Linear/Bias:0: (100,)
2016-11-25 11:15:48,107 INFO   Net/RNN/OutputProjectionWrapper/Linear/Matrix:0: (100, 10000)
2016-11-25 11:15:48,107 INFO   Net/RNN/OutputProjectionWrapper/Linear/Bias:0: (10000,)
"""

if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="rnn", help="Name of the run used in saving logs and models")
    args = parser.parse_args()

    os.makedirs(os.path.join(LOG_DIR, args.name), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, args.name), exist_ok=True)

    config = Config()
    with tf.variable_scope('RNNLM') as scope:
        model = RNNLM_Model(config)

    summ_perpl_train_t = tf.placeholder(tf.float32, name='perplexity_train')
    tf.scalar_summary("perplexity_train", summ_perpl_train_t)

    summ_perpl_val_t = tf.placeholder(tf.float32, name='perplexity_val')
    tf.scalar_summary("perplexity_val", summ_perpl_val_t, collections=['summary_epoch'])

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    summaries = tf.merge_all_summaries()
    summaries_epoch = tf.merge_all_summaries('summary_epoch')

    with tf.Session() as session:
        saver = tf.train.Saver(max_to_keep=16)
        writer = tf.train.SummaryWriter(os.path.join(LOG_DIR, args.name), session.graph)

        def train_summary(step, perpl):
            s, = session.run([summaries], feed_dict={
                summ_perpl_train_t: perpl
            })
            writer.add_summary(s, global_step=step)
            writer.flush()

        session.run(init)
        show_variables()
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            ###
            train_pp = model.run_epoch(
                session, model.encoded_train,
                train_op=model.train_step, summary_hook=train_summary)
            valid_pp = model.run_epoch(session, model.encoded_valid)

            s, = session.run([summaries_epoch], feed_dict={
                summ_perpl_val_t: valid_pp
            })
            writer.add_summary(s, global_step=model.global_step)
            writer.flush()

            print('Training perplexity: {}'.format(train_pp))
            print('Validation perplexity: {}'.format(valid_pp))
            saver.save(session, os.path.join(SAVE_DIR, args.name, "model-epoch=%d" % epoch))


if __name__ == "__main__1":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="rnn", help="Name of the run used in saving logs and models")
    parser.add_argument("--max-epoch", type=int, default=16,
                        help="If specified, stop after given amount of epoches, default=16")
    args = parser.parse_args()

    os.makedirs(os.path.join(LOG_DIR, args.name), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, args.name), exist_ok=True)

    log.info("Loading PTB dataset...")
#    data = ptb.PTBDataset("data", vocab.Vocab(), num_steps=10)
#    data.load_dataset()
#    log.info("Loaded: %s", data)
    data = Data()

    with tf.Session() as session:
        ph_labels = tf.placeholder(tf.int32, shape=(None, NUM_STEPS), name="labels")

        ph_input, initial_state, outputs, final_state = make_net(data.vocab.size())

#        targets = tf.split(split_dim=1, num_split=NUM_STEPS, value=ph_labels)
        output = tf.reshape(tf.concat(1, outputs), [-1, data.vocab.size()])
        log.info("Loss info:")
        log.info("Output: %s", output)
        labels = tf.reshape(ph_labels, [-1])
        log.info("Labels: %s", labels)
        loss_t = sequence_loss([output], [labels], [tf.ones([BATCH * NUM_STEPS])])
        tf.add_to_collection('total_loss', loss_t)
        loss = tf.add_n(tf.get_collection('total_loss'))

# 2016-11-25 12:18:13,122 INFO Loss info:
# 2016-11-25 12:18:13,122 INFO Output: Tensor("Reshape:0", shape=(640, 10000), dtype=float32)
# 2016-11-25 12:18:13,123 INFO Labels: Tensor("Reshape_1:0", shape=(?,), dtype=int32)

        opt = tf.train.AdamOptimizer(LR)
        opt_t = opt.minimize(loss)

        # summaries
        writer = tf.train.SummaryWriter(os.path.join(LOG_DIR, args.name), session.graph)
        summ_perpl_train_t = tf.placeholder(tf.float32, name='perplexity_train')
        tf.scalar_summary("perplexity_train", summ_perpl_train_t)

        summ_perpl_val_t = tf.placeholder(tf.float32, name='perplexity_val')
        tf.scalar_summary("perplexity_val", summ_perpl_val_t, collections=['summary_epoch'])

        saver = tf.train.Saver(max_to_keep=args.max_epoch)

        summaries = tf.merge_all_summaries()
        summaries_epoch = tf.merge_all_summaries('summary_epoch')
        session.run(tf.initialize_all_variables())

        show_variables()
        global_step = 0
        epoch = 0
        progress = 0.0
        while args.max_epoch is None or args.max_epoch > epoch:
            losses = []
            state = initial_state.eval()
            for train_x, train_y in ptb_iterator(data.encoded_train, BATCH, NUM_STEPS):
                loss, state, _ = session.run([loss_t, final_state, opt_t], feed_dict={
                    ph_input: train_x,
                    ph_labels: train_y,
                    initial_state: state
                })
                losses.append(loss)
                if global_step % 100 == 0:
                    m_perpl = np.exp(np.mean(losses))
                    log.info("Epoch=%d, iter=%d, epoch_perc=%.2f%%, perplexity=%s",
                             epoch, global_step, progress*100.0, m_perpl)
                    summ_res, = session.run([summaries], feed_dict={
                        summ_perpl_train_t: m_perpl,
                    })
                    writer.add_summary(summ_res, global_step)
                    writer.flush()
                    losses = []
                global_step += 1
            saver.save(session, os.path.join(SAVE_DIR, args.name, "model-epoch=%d" % epoch))

            # validation
            log.info("Running validation...")
            losses = []
            for x, y in ptb_iterator(data.encoded_valid, BATCH, NUM_STEPS):
                loss, = session.run([loss_t], feed_dict={
                    ph_input: x,
                    ph_labels: y
                })
                losses.append(loss)
            m_perpl = np.exp(np.mean(losses))
            summ_res, = session.run([summaries_epoch], feed_dict={
                summ_perpl_val_t: m_perpl
            })
            writer.add_summary(summ_res, global_step)
            writer.flush()
            log.info("Validiation perplexity: %s", m_perpl)
            epoch += 1

        log.info("Running test...")
        losses = []
        for x, y in ptb_iterator(data.encoded_test, BATCH, NUM_STEPS):
            loss, = session.run([loss_t], feed_dict={
                ph_input: x,
                ph_labels: y
            })
            losses.append(loss)
        log.info("Test perplexity: %s", np.exp(np.mean(losses)))

    pass
