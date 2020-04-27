#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_dense.py [options]
Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format)
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format)
    --log_dir NAME           log dir name
    --data_dir NAME          data dir name
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --evaluate               example evaluation mode using a restored model
"""

from typing import Sequence, Any
from collections import defaultdict
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import json

from util import glorot_init

from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv1D, MaxPooling1D, Conv2D, GlobalMaxPooling2D, MaxPooling2D, Convolution2D
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.utils import np_utils
from keras import backend as K

from keras.layers import Conv1D, Dense, MaxPool1D, concatenate, Flatten,Multiply
from keras import Input, Model
from keras.utils import plot_model


'''
Comments provide the expected tensor shapes where helpful.
Key to symbols in comments:
---------------------------
[...]:  a tensor
; ; :   a list
b:      batch size
e:      number of edge types (4)
v:      number of vertices per graph in this batch
h:      GNN hidden size
'''

class DenseGGNNModel():
    def __init__(self, opt):
        self.batch_size = opt.batch_size
        self.node_token_dim = opt.node_token_dim
        self.node_type_dim = opt.node_type_dim
        self.node_dim = self.node_type_dim + self.node_token_dim
        self.label_dim = self.node_type_dim + self.node_token_dim

        self.node_type_lookup = opt.node_type_lookup
        self.node_token_lookup = opt.node_token_lookup

        self.hidden_layer_size = opt.hidden_layer_size
        self.num_hidden_layer = opt.num_hidden_layer
        self.aggregation_type = opt.aggregation
        self.distributed_function = opt.distributed_function
        self.num_labels = opt.num_labels
        self.num_edge_types = opt.n_edge_types
        self.num_timesteps= opt.n_steps
        self.placeholders = {}
        self.weights = {}

        self.prepare_specific_graph_model()
        self.nodes_representation = self.compute_nodes_representation()

        initial_nodes_representation = tf.concat([self.node_type_representations, self.node_token_representations], -1)

        self.initial_nodes_representation = initial_nodes_representation

        features = self.aggregation_layer(self.nodes_representation, self.initial_nodes_representation)
        self.logits = tf.reduce_mean(features,1)
        self.loss = self.loss_layer(self.logits)
        self.prediction = tf.round(tf.nn.sigmoid(self.logits))

    def prepare_specific_graph_model(self) -> None:
        node_dim = self.node_dim

        # initializer = tf.contrib.layers.xavier_initializer()
        # inputs
        # self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        # self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')
        self.node_type_embeddings = tf.Variable(glorot_init([len(self.node_type_lookup.keys()), self.node_type_dim]), name='node_type_embeddings')
        self.node_token_embeddings = tf.Variable(glorot_init([len(self.node_token_lookup.keys()), self.node_token_dim]), name='node_token_embeddings')
        self.label_embeddings = tf.Variable(glorot_init([2, self.label_dim]), name='label_embeddings')

        self.placeholders["node_type_indices"] = tf.placeholder(tf.int32, shape=[None,None], name='node_type_indices')
        self.placeholders["node_token_indices"] = tf.placeholder(tf.int32, shape=[None,None,None], name='node_token_indices')

        self.node_type_representations = tf.nn.embedding_lookup(self.node_type_embeddings, self.placeholders["node_type_indices"])
        self.node_token_representations = tf.nn.embedding_lookup(self.node_token_embeddings, self.placeholders["node_token_indices"])
        self.node_token_representations = tf.reduce_mean(self.node_token_representations, axis=2)

        # self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, None, self.node_dim], name='node_features')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, (),  name='num_vertices')
        # self.placeholders['labels'] = tf.placeholder(tf.int32, shape=[None,30], name='labels')
        self.placeholders['labels'] = tf.placeholder(tf.float32, (None, 1))

        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,[None, self.num_edge_types, None, None], name='adjacency_matrix')     # [b, e, v, v]
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])                    # [e, b, v, v]

        # batch normalization
        self.placeholders['is_training'] = tf.placeholder(tf.bool, name="is_train")
        self.node_type_representations = tf.layers.batch_normalization(self.node_type_representations, training=self.placeholders['is_training'])
        self.node_token_representations = tf.layers.batch_normalization(self.node_token_representations, training=self.placeholders['is_training'])

        # weights
        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, node_dim, node_dim]),name='edge_weights')
        self.weights['edge_biases'] = tf.Variable(tf.zeros([self.num_edge_types, 1, node_dim]),name='edge_biases')

        self.xavier_initializer = tf.contrib.layers.xavier_initializer()
        # self.weights["hidden_layer_weights"] = tf.Variable(xavier_initializer([self.node_dim, self.num_labels]), name='hidden_layer_weights')
        # self.weights["hidden_layer_biases"] = tf.Variable(xavier_initializer([self.num_labels,]), name='hidden_layer_biases')

        self.weights['attention_weights'] = tf.Variable(glorot_init([self.node_dim,1]),name='attention_weights')


        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(node_dim)
            # cell = tf.python.ops.rnn_cell.GRUCell(node_dim)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
            # cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell

    def compute_nodes_representation(self):
        node_dim = self.node_dim
        v = self.placeholders['num_vertices']
        # h = self.placeholders['initial_node_representation']                                                # [b, v, h]

        h = tf.concat([self.node_type_representations, self.node_token_representations], -1)
        h = tf.reshape(h, [-1, self.node_token_dim + self.node_type_dim])

        with tf.compat.v1.variable_scope("gru_scope") as scope:
            for i in range(self.num_timesteps):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    # print("edge type : " + str(edge_type))
                    # m = tf.matmul(h, tf.nn.dropout(self.weights['edge_weights'][edge_type], rate=1-self.placeholders['edge_weight_dropout_keep_prob'])) # [b*v, h]
                    m = tf.matmul(h, self.weights['edge_weights'][edge_type])                               # [b*v, h]

                    m = tf.reshape(m, [-1, v, node_dim])                                                       # [b, v, h]
                    m += self.weights['edge_biases'][edge_type]                                             # [b, v, h]

                    if edge_type == 0:
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)
                acts = tf.reshape(acts, [-1, node_dim])                                                        # [b*v, h]

                h = self.weights['node_gru'](acts, h)[1]                                                    # [b*v, h]
            last_h = tf.reshape(h, [-1, v, node_dim])
        return last_h

    def aggregation_layer(self, nodes_representation,initial_nodes_representation):

        input1_ = nodes_representation
        input2_ = tf.concat([nodes_representation,initial_nodes_representation], -1)



        x1 = Conv1D(
            data_format='channels_last',
            filters=1,
            kernel_size=3
        )(input1_)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling1D(3, 2, padding='same')(x1)
        x1 = Conv1D(
            data_format='channels_last',
            filters=1,
            kernel_size=1
        )(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling1D(2, 2, padding='same')(x1)
        x1 = Dense(units=1)(x1)

        x2 = Conv1D(
            data_format='channels_last',
            filters=1,
            kernel_size=3
        )(input2_)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling1D(3, 2, padding='same')(x2)
        x2 = Conv1D(
            data_format='channels_last',
            filters=1,
            kernel_size=1
        )(x2)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling1D(2, 2, padding='same')(x2)
        x2 = Dense(units=1)(x2)
        x = Multiply()([x1,x2])
        # x = tf.reduce_mean(x,1)
        # x = Activation('sigmoid')(x)
        output_ = x
        return output_

    def loss_layer(self, logits_node):
        """Create a loss layer for training."""
        labels = self.placeholders['labels']

        with tf.name_scope('loss_layer'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits_node, name='cross_entropy'
            )

            loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
            return loss