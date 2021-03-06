'''
Class Model: model for the deep clustering speech seperation
'''
import numpy as np
import tensorflow as tf

from .constant import *


class Model(object):
    def __init__(self, n_hidden, batch_size, p_keep_ff, p_keep_rc):
        '''n_hidden: number of hidden states
           p_keep_ff: forward keep probability
           p_keep_rc: recurrent keep probability'''
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.p_keep_ff = p_keep_ff
        self.p_keep_rc = p_keep_rc
        # biases and weights for the last layer
        self.weights = {
            'out': tf.Variable(
                tf.random_normal([2 * n_hidden, EMBBEDDING_D * NEFF], mean=0.0, stddev=1.0))
        }
        self.biases = {
            'out': tf.Variable(
                tf.random_normal([EMBBEDDING_D * NEFF], mean=0.0, stddev=1.0))
        }

    def inference(self, x):
        '''The structure of the network'''
        
        state_concate = x

        for i in range(N_LAYERS):
            with tf.variable_scope('BLSTM%d' % (i + LAYER_NAME_OFFSET)) as scope:
                lstm_fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    self.n_hidden, layer_norm=False,
                    dropout_keep_prob=self.p_keep_rc)
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_fw_cell, input_keep_prob=1,
                    output_keep_prob=self.p_keep_ff)
                lstm_bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    self.n_hidden, layer_norm=False,
                    dropout_keep_prob=self.p_keep_rc)
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_bw_cell, input_keep_prob=1,
                    output_keep_prob=self.p_keep_ff)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    lstm_fw_cell, lstm_bw_cell, state_concate,
                    sequence_length=[FRAMES_PER_SAMPLE] * self.batch_size,
                    dtype=tf.float32)
                state_concate = tf.concat(outputs, 2)

        # one layer of embedding output with tanh activation function
        out_concate = tf.reshape(state_concate, [-1, self.n_hidden * 2])
        emb_out = tf.matmul(out_concate,
                            self.weights['out']) + self.biases['out']
        emb_out = tf.nn.tanh(emb_out)
        reshaped_emb = tf.reshape(emb_out, [-1, NEFF, EMBBEDDING_D])
        # normalization before output
        normalized_emb = tf.nn.l2_normalize(reshaped_emb, 2)
        return normalized_emb

    def loss(self, embeddings, Y):
        '''Defining the loss function'''
        embeddings_rs = tf.reshape(embeddings, shape=[-1, EMBBEDDING_D])
        embeddings_v = tf.reshape(
            embeddings_rs, [-1, FRAMES_PER_SAMPLE * NEFF, EMBBEDDING_D])
        # get the Y(speaker indicator function)
        Y_rs = tf.reshape(Y, shape=[-1, 2])
        Y_v = tf.reshape(Y_rs, shape=[-1, FRAMES_PER_SAMPLE * NEFF, 2])
        # fast computation format of the embedding loss function
        loss_batch = tf.nn.l2_loss(
            tf.matmul(tf.transpose(
                embeddings_v, [0, 2, 1]), embeddings_v)) - \
            2 * tf.nn.l2_loss(
                tf.matmul(tf.transpose(
                    embeddings_v, [0, 2, 1]), Y_v)) + \
            tf.nn.l2_loss(
                tf.matmul(tf.transpose(
                    Y_v, [0, 2, 1]), Y_v))
        loss_v = (loss_batch) / self.batch_size / (FRAMES_PER_SAMPLE^2)
        return loss_v

    def train(self, loss, lr):
        '''Optimizer'''
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 200)
        train_op = optimizer.apply_gradients(
            zip(gradients, v))
        return train_op
