#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2: Recurrent neural nets for NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from ner_model import NERModel
from defs import LBLS
from rnn_cell import RNNCell

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_word_features = 2 # Number of features for every word in the input.
    window_size = 1
    n_features = (2 * window_size + 1) * n_word_features # Number of features for every word in the input.
    max_length = 120 # longest sequence to parse
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 300
    batch_size = 32
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001

    def __init__(self, args):
        self.cell = args.cell

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"
        if "summarize" in args:
            self.summarize = args.summarize
        elif "verbose" in args:
            self.summarize = args.verbose
        else:
            self.summarize = False


def pad_sequences(data, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    TODO: In the code below, for every sentence, labels pair in @data,
    (a) create a new sentence which appends zero feature vectors until
    the sentence is of length @max_length. If the sentence is longer
    than @max_length, simply truncate the sentence to be @max_length
    long.
    (b) create a new label sequence similarly.
    (c) create a _masking_ sequence that has a True wherever there was a
    token in the original sequence, and a False for every padded input.

    Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
    0, 0], and max_length = 5, we would construct
        - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
        - a new label seqeunce: [1, 0, 0, 4, 4], and
        - a masking seqeunce: [True, True, True, False, False].

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels. 
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sentence', labels', mask).
        Each of sentence', labels' and mask are of length @max_length.
        See the example above for more details.
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_features
    zero_label = 4 # corresponds to the 'O' tag

    for sentence, labels in data:
        ### YOUR CODE HERE (~4-6 lines)
        padding_length = max_length - len(sentence) if max_length - len(sentence) > 0 else 0
        mask = [True] * len(sentence)
        new_sentence = sentence + (padding_length * [zero_vector])
        new_labels = labels + (padding_length * [zero_label])
        new_mask = mask + (padding_length * [False])
        if len(new_sentence) > max_length:
            new_sentence = new_sentence[:max_length]
            new_labels = new_labels[:max_length]
            new_mask = new_mask[:max_length]
        ret.append((new_sentence, new_labels, new_mask))
        ### END YOUR CODE ###
    return ret


class RNNModel(NERModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        TODO: Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.mask_placeholder
            self.dropout_placeholder

        HINTS:
            - Remember to use self.max_length NOT Config.max_length

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~4-6 lines)
        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length, self.config.n_features), name="input_placeholder")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.max_length), name="labels_placeholder")
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.max_length), name="mask_placeholder")
        self.dropout_placeholder = tf.placeholder(tf.float32, shape=(), name="dropout_placeholder")
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1.0):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE (~6-10 lines)
        feed_dict = {self.input_placeholder: inputs_batch,
                     self.mask_placeholder: mask_batch,
                     self.dropout_placeholder: dropout}
        # add labels batch to feed dict only if it is not None
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        ### YOUR CODE HERE (~4-6 lines)
        embeddings_lookup = tf.nn.embedding_lookup(tf.Variable(self.pretrained_embeddings), self.input_placeholder)
        embeddings = tf.reshape(embeddings_lookup,
                                (-1, self.max_length, self.config.n_features * self.config.embed_size))
        ### END YOUR CODE
        return embeddings

    def add_prediction_op_rnn(self, x, dropout_rate):
        """Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2
    
        TODO: There a quite a few things you'll need to do in this function:
            - Define the variables U, b_2.
            - Define the vector h as a constant and inititalize it with
              zeros. See tf.zeros and tf.shape for information on how
              to initialize this variable to be of the right shape.
              https://www.tensorflow.org/api_docs/python/tf/zeros
              https://www.tensorflow.org/api_docs/python/tf/shape
            - In a for loop, begin to unroll the RNN sequence. Collect
              the predictions in a list.
            - When unrolling the loop, from the second iteration
              onwards, you will HAVE to call
              tf.get_variable_scope().reuse_variables() so that you do
              not create new variables in the RNN cell.
              See https://www.tensorflow.org/api_guides/python/state_ops#Sharing_Variables
            - Concatenate and reshape the predictions into a predictions
              tensor.
        Hint: You will find the function tf.stack (similar to np.asarray)
              useful to assemble a list of tensors into a larger tensor.
              https://www.tensorflow.org/api_docs/python/tf/stack
        Hint: You will find the function tf.transpose and the perms
              argument useful to shuffle the indices of the tensor.
              https://www.tensorflow.org/api_docs/python/tf/transpose
    
        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder
    
        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """

        cell = RNNCell(Config.n_features * Config.embed_size, Config.hidden_size)
        preds = []  # Predicted output at each timestep should go here!

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        ### YOUR CODE HERE (~4-6 lines)
        U = tf.get_variable("U", shape=(self.config.hidden_size, self.config.n_classes),
                            initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.Variable(tf.zeros(self.config.n_classes))
        state = tf.zeros((1, self.config.hidden_size))
        ### END YOUR CODE

        with tf.variable_scope("RNN"):
            for time_step in range(self.max_length):
                ### YOUR CODE HERE (~6-10 lines)
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                o_t, h_t = cell(x[:, time_step, :], state)  # each cell handles different timestamp
                o_drop_t = tf.nn.dropout(o_t, dropout_rate)
                y_t = tf.matmul(o_drop_t, U) + b2
                preds.append(y_t)
                ### END YOUR CODE

        # Make sure to reshape @preds here.
        ### YOUR CODE HERE (~2-4 lines)
        preds = tf.stack(preds)  # stack takes list of tensors and create one tensor from them
        preds = tf.transpose(preds, (1, 0, 2))  # dim0->dim1, dim1->dim0, dim2->dim2 re-arange the shape from(max_length, -1, 5) to(-1, max_length, 5)
        ### END YOUR CODE

        return preds

    def add_prediction_op_gru(self, x, dropout_rate):
        """Adds a dynamic RNN with GRU cell, followed by a single layer feed-forward neural network.
        TODO: The following steps should be made using TensorFlow API functions only,
              meaning that you should not create any variables explicitly.
            - Create a GRU cell of the same hidden size as before
            - Apply a dropout
            - Create a dynamic RNN
            - Transfer the RNN outputs though a single layer feed-forward neural network.
              The function tf.layers.dense can be useful here. 
              https://www.tensorflow.org/api_docs/python/tf/layers/dense

        Remember:
            * Use the xavier initilization for matrices, and constant initializer for the bias.
            * The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        ### YOUR CODE HERE (~4-6 lines)
        # Create layer og GRU cells
        gru = tf.contrib.rnn.GRUCell(self.max_length, name="gru")
        gru = tf.contrib.rnn.DropoutWrapper(gru, output_keep_prob=dropout_rate)
        # tensorflow rnn
        output, state = tf.nn.dynamic_rnn(gru, x, dtype=tf.float32)
        # apply FFNN on the outputs
        preds = tf.layers.dense(inputs=output, units=self.config.n_classes,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
        ### END YOUR CODE

        return preds

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """

        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        if self.config.cell == "rnn":
            preds = self.add_prediction_op_rnn(x, dropout_rate)
        elif self.config.cell == "gru":
            preds = self.add_prediction_op_gru(x, dropout_rate)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        assert preds.get_shape().as_list() == [None, self.max_length, self.config.n_classes],\
            "predictions are not of the right shape. Expected {}, got {}".format(
                [None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        TODO: Compute averaged cross entropy loss for the predictions.
        Importantly, you must ignore the loss for any masked tokens.

        Hint: You might find tf.boolean_mask useful to mask the losses on masked tokens.
        Hint: You can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-4 lines)
        masked_preds = tf.boolean_mask(preds, self.mask_placeholder)
        masked_labels = tf.boolean_mask(self.labels_placeholder, self.mask_placeholder)
        loss_flat = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=masked_preds, labels=masked_labels)
        loss = tf.reduce_mean(loss_flat)
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        ### END YOUR CODE
        return train_op

    def add_summary_op(self, pred, loss):
        """Sets up the summary Op.

        Generates summaries about the model to be displayed by TensorBoard.
        https://www.tensorflow.org/api_docs/python/tf/summary
        for more information.
        
        TODO: Append to the 'records' list 3 summaries:
            - add histogram summary of the prediction logits (pred)
            - a scalar summary of the average loss (loss)
            - a scalar summary of the average prediction entropy:
                - transform the prediction logits tensors into probability tensors, by directly applying softmax. 
                  You can use the function tf.nn.softmax
                    https://www.tensorflow.org/api_docs/python/tf/nn/softmax
                - store the results in "self.probs" (self.probs = ...)
                - use these probabilities to calculate the entropy, also use the function tf.log
                    https://www.tensorflow.org/api_docs/python/tf/log
            
        Remember: clip the probability values before applying the logarithm function, with
                  the function tf.clip_by_value.
                    https://www.tensorflow.org/api_docs/python/tf/clip_by_value
        Hint: You might find tf.boolean_mask useful.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
            loss: Loss tensor (a scalar).
        Returns:
            summary: training records summary.
        """
        records = []

        ### YOUR CODE HERE (~5-10 lines)
        pred = tf.boolean_mask(pred, self.mask_placeholder)

        records.append(tf.summary.histogram('prediction_logits', pred))
        mean_loss = tf.reduce_mean(loss)
        records.append(tf.summary.scalar('mean_loss', mean_loss))
        self.probs = tf.nn.softmax(pred)
        entropy = -tf.reduce_sum(self.probs * tf.log(tf.clip_by_value(self.probs, 1e-10, 1.0)))
        records.append(tf.summary.scalar('predictions_entropy', entropy))
        ### END YOUR CODE

        assert hasattr(self, 'probs'), "self.probs should be set."
        summary = tf.summary.merge(records)
        return summary

    def preprocess_sequence_data(self, examples):
        def featurize_windows(data, start, end, window_size = 1):
            """Uses the input sequences in @data to construct new windowed data points.
            """
            ret = []
            for sentence, labels in data:
                from util import window_iterator
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(sum(window, []))
                ret.append((sentence_, labels))
            return ret

        examples = featurize_windows(examples, self.helper.START, self.helper.END)
        return pad_sequences(examples, self.max_length)

    def consolidate_predictions(self, examples_raw, examples, preds, probs):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)
        if probs is not None:
            assert len(examples_raw) == len(probs)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels)
            if probs is not None:
                probs_ = [l for l, m in zip(probs[i], mask) if m]  # only select elements of mask.
                assert len(probs_) == len(labels)
                ret.append([sentence, labels, labels_, probs_])
            else:
                ret.append([sentence, labels, labels_])

        return ret

    def predict_on_batch(self, sess, inputs_batch, mask_batch, summarize=False):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        if summarize:
            predictions, confidence = sess.run([tf.argmax(self.pred, axis=2), tf.reduce_max(self.probs, axis=2)], feed_dict=feed)
            return predictions, confidence
        else:
            predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
            return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch, summarize=False):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        if summarize:
            _, loss, summary = sess.run([self.train_op, self.loss, self.summary], feed_dict=feed)
            return loss, summary
        else:
            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
            return loss

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(RNNModel, self).__init__(helper, config, report)
        self.max_length = min(Config.max_length, helper.max_length)
        Config.max_length = self.max_length # Just in case people make a mistake.
        self.pretrained_embeddings = pretrained_embeddings

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build(config.summarize)


def test_pad_sequences():
    Config.n_features = 2
    data = [
        ([[4,1], [6,0], [7,0]], [1, 0, 0]),
        ([[3,0], [3,4], [4,5], [5,3], [3,4]], [0, 1, 0, 2, 3]),
        ]
    ret = [
        ([[4,1], [6,0], [7,0], [0,0]], [1, 0, 0, 4], [True, True, True, False]),
        ([[3,0], [3,4], [4,5], [5,3]], [0, 1, 0, 2], [True, True, True, True])
        ]

    ret_ = pad_sequences(data, 4)
    assert len(ret_) == 2, "Did not process all examples: expected {} results, but got {}.".format(2, len(ret_))
    for i in range(2):
        assert len(ret_[i]) == 3, "Did not populate return values corrected: expected {} items, but got {}.".format(3, len(ret_[i]))
        for j in range(3):
            assert ret_[i][j] == ret[i][j], "Expected {}, but got {} for {}-th entry of {}-th example".format(ret[i][j], ret_[i][j], j, i)


def do_test1(_):
    logger.info("Testing pad_sequences")
    test_pad_sequences()
    logger.info("Passed!")


def do_test2(args):
    logger.info("Testing implementation of RNNModel")
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)

    logger.info("Model did not crash!")
    logger.info("Passed!")


def do_train(args):
    # Set up some parameters.
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(config.output_path) if args.summarize else None

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev, writer)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                # Save predictions in a text file.
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                predictions = [[LBLS[l] for l in preds] for preds in predictions]
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_conll(f, output)
                with open(model.config.eval_output, 'w') as f:
                    for sentence, labels, predictions in output:
                        print_sentence(f, sentence, labels, predictions)


def do_evaluate(args):
    config = Config(args)
    helper = ModelHelper.load(args.model_path)
    input_data = read_conll(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            probs = None
            for output in model.output(session, input_data, summarize=args.verbose):
                if args.verbose:
                    (sentence, labels, predictions, probs) = output
                else:
                    (sentence, labels, predictions) = output
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions, probs)


def do_shell(args):
    config = Config(args)
    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)

            print("""Welcome!
You can use this shell to explore the behavior of your model.
Please enter sentences with spaces between tokens, e.g.,
input> Germany 's representative to the European Union 's veterinary committee .
""")
            while True:
                # Create simple REPL
                try:
                    sentence = raw_input("input> ")
                    tokens = sentence.strip().split(" ")
                    probs = None
                    for output in model.output(session, [(tokens, ["O"] * len(tokens))], summarize=args.verbose):
                        if args.verbose:
                            (sentence, _, predictions, probs) = output
                        else:
                            (sentence, _, predictions) = output
                        predictions = [LBLS[l] for l in predictions]
                        print_sentence(sys.stdout, sentence, [""] * len(tokens), predictions, probs)
                except EOFError:
                    print("Closing session.")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test1', help='')
    command_parser.set_defaults(func=do_test1)

    command_parser = subparsers.add_parser('test2', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/tiny.conll", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/tiny.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_test2)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/train.conll", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/dev.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument('-s', '--summarize', action='store_true', default=False, help='Export tensor summaries')
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="data/dev.conll", help="Training data")
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    command_parser.add_argument('-s', '--verbose', action='store_true', default=False, help='Display prediction probabilities')
    command_parser.set_defaults(func=do_evaluate)

    command_parser = subparsers.add_parser('shell', help='')
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.add_argument('-s', '--verbose', action='store_true', default=False, help='Display prediction probabilities')
    command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)