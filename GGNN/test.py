import argparse
import random

import pickle

import tensorflow as tf
from dataset import MethodNamePredictionData
from util import ThreadedIterator
from dense_ggnn import DenseGGNNModel
import os
import sys
import re
import time

from bidict import bidict
import copy
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import evaluation
from scipy.spatial import distance
from datetime import datetime
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv1D, MaxPooling1D, Conv2D, GlobalMaxPooling2D, MaxPooling2D, Convolution2D
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.utils import np_utils
from keras import backend as K

from keras.layers import Conv1D, Dense, MaxPool1D, concatenate, Flatten
from keras import Input, Model
from keras.utils import plot_model
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int,
                    default=128, help='input batch size')
parser.add_argument('--train_batch_size', type=int,
                    default=128, help='train input batch size')
parser.add_argument('--test_batch_size', type=int,
                    default=128, help='test input batch size')
parser.add_argument('--val_batch_size', type=int,
                    default=128, help='val input batch size')
parser.add_argument('--state_dim', type=int, default=30,
                    help='GGNN hidden state dimension size')
parser.add_argument('--node_type_dim', type=int, default=50,
                    help='node type dimension size')
parser.add_argument('--node_token_dim', type=int,
                    default=100, help='node token dimension size')
parser.add_argument('--hidden_layer_size', type=int,
                    default=100, help='size of hidden layer')
parser.add_argument('--num_hidden_layer', type=int,
                    default=1, help='number of hidden layer')
parser.add_argument('--n_steps', type=int, default=6,
                    help='propagation steps number of GGNN')
parser.add_argument('--n_edge_types', type=int, default=7,
                    help='number of edge types')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--cuda', default="2", type=str, help='enables cuda')
parser.add_argument('--verbal', type=bool, default=True,
                    help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--model_path', default="model",
                    help='path to save the model')
# parser.add_argument('--model_accuracy_path', default="model_accuracy/method_name.txt",
#                     help='path to save the the best accuracy of the model')
parser.add_argument('--n_hidden', type=int, default=50,
                    help='number of hidden layers')
parser.add_argument('--log_path', default="logs/",
                    help='log path for tensorboard')
parser.add_argument('--checkpoint_every', type=int,
                    default=10, help='check point to save model')
parser.add_argument('--validating', type=int,
                    default=1, help='validating or not')
parser.add_argument('--graph_size_threshold', type=int,
                    default=500, help='graph size threshold')
parser.add_argument('--sampling_size', type=int,
                    default=1, help='sampling size for each epoch')
parser.add_argument('--best_f1', type=float,
                    default=0.0, help='best f1 to save model')
parser.add_argument('--aggregation', type=int, default=1, choices=range(0, 4),
                    help='0 for max pooling, 1 for attention with sum pooling, 2 for attention with max pooling, 3 for attention with average pooling')
parser.add_argument('--distributed_function', type=int, default=0,
                    choices=range(0, 2), help='0 for softmax, 1 for sigmoid')
parser.add_argument('--train_path', default="./train",
                    help='path of training data')
parser.add_argument('--val_path', default="./val",
                    help='path of validation data')
parser.add_argument('--test_path', default="./test_graph",
                    help='path of test data')
parser.add_argument('--dataset', default="my_dataset",
                    help='name of dataset')
parser.add_argument('--node_type_vocabulary_path', default="./nodetype.txt",
                    help='name of dataset')
parser.add_argument('--token_vocabulary_path', default="./token.txt",
                    help='name of dataset')
parser.add_argument('--train_label_vocabulary_path', default="preprocessed_data/train_label_vocab.txt",
                    help='name of dataset')
parser.add_argument('--val_label_vocabulary_path', default="preprocessed_data/val_label_vocab.txt",
                    help='name of dataset')
parser.add_argument('--task', type=int, default=1,
                    choices=range(0, 2), help='0 for training, 1 for testing')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

print(opt)


def load_vocabs(opt):
    node_type_lookup = {}
    node_token_lookup = {}

    node_type_vocabulary_path = opt.node_type_vocabulary_path
    token_vocabulary_path = opt.token_vocabulary_path

    with open(node_type_vocabulary_path, "r") as f2:
        data = f2.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_type_lookup[splits[1]] = int(splits[0])

    with open(token_vocabulary_path, "r") as f3:
        data = f3.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_token_lookup[splits[1]] = int(splits[0])

    node_type_lookup = bidict(node_type_lookup)
    node_token_lookup = bidict(node_token_lookup)
    return node_type_lookup, node_token_lookup


def form_model_path(opt):
    model_traits = {}
    model_traits["dataset"] = str(opt.dataset)
    model_traits["aggregation"] = str(opt.aggregation)
    model_traits["distributed_function"] = str(opt.distributed_function)
    model_traits["node_type_dim"] = str(opt.node_type_dim)
    model_traits["node_token_dim"] = str(opt.node_token_dim)

    model_path = []
    for k, v in model_traits.items():
        model_path.append(k + "_" + v)

    return "vul_prediction" + "_" + "-".join(model_path)


def get_best_f1_score(opt):
    best_f1_score = 0.0

    try:
        os.mkdir("model_accuracy")
    except Exception as e:
        print(e)

    opt.model_accuracy_path = os.path.join("model_accuracy", form_model_path(opt) + ".txt")

    if os.path.exists(opt.model_accuracy_path):
        print("Model accuracy path exists : " + str(opt.model_accuracy_path))
        with open(opt.model_accuracy_path, "r") as f4:
            data = f4.readlines()
            for line in data:
                best_f1_score = float(line.replace("\n", ""))
    else:
        print("Creating model accuracy path : " + str(opt.model_accuracy_path))
        with open(opt.model_accuracy_path, "w") as f5:
            f5.write("0.0")

    return best_f1_score

def main(opt):

    node_type_lookup, node_token_lookup = load_vocabs(opt)
    opt.num_labels = 2
    opt.node_type_lookup = node_type_lookup
    opt.node_token_lookup = node_token_lookup


    test_opt = copy.deepcopy(opt)
    test_opt.num_labels = 2
    test_opt.node_token_lookup = node_token_lookup
    test_dataset = MethodNamePredictionData(test_opt, opt.test_path, False, False, True)

    ggnn = DenseGGNNModel(opt)

    # For debugging purpose
    initial_nodes_representation = ggnn.initial_nodes_representation
    logits = ggnn.logits
    loss_node = ggnn.loss
    prediction = ggnn.prediction

    optimizer = tf.compat.v1.train.AdamOptimizer(opt.lr)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        training_point = optimizer.minimize(loss_node)

    saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)
    init = tf.global_variables_initializer()
    best_f1_score = get_best_f1_score(opt)
    print("Best f1 score : " + str(best_f1_score))

    opt.model_path = "model/vul_prediction_distributed_function_0-node_token_dim_100-dataset_my_dataset-aggregation_1-node_type_dim_50"
    checkfile = os.path.join(opt.model_path, 'cnn_tree.ckpt')
    ckpt = tf.train.get_checkpoint_state(opt.model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Using the trained model : " + str(checkfile))
    else:
        print("no trained model in :", checkfile)

    with tf.Session() as sess:
        sess.run(init)
        if opt.task == 1:
            print("Test model.............")
            print(checkfile)
            if ckpt and ckpt.model_checkpoint_path:
                print("Checkpoint path : " + str(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
                for i, var in enumerate(saver._var_list):
                    print('Var {}: {}'.format(i, var))

            average_f1 = 0.0
            test_batch_iterator = ThreadedIterator(
            test_dataset.make_minibatch_iterator(), max_queue_size=5)

            all_predicted_labels = []
            all_ground_truth_labels = []
            for test_step, test_batch_data in enumerate(test_batch_iterator):
                print("----------------------------------------")
                scores,predicted = sess.run(
                    [logits,prediction],
                    feed_dict={
                        ggnn.placeholders["num_vertices"]: test_batch_data["num_vertices"],
                        ggnn.placeholders["adjacency_matrix"]: test_batch_data['adjacency_matrix'],
                        ggnn.placeholders["node_type_indices"]: test_batch_data["node_type_indices"],
                        ggnn.placeholders["node_token_indices"]: test_batch_data["node_token_indices"],
                        ggnn.placeholders["is_training"]: False
                    }
                )
                predicted_labels = predicted

                ground_truth_labels = test_batch_data['labels']

                f1_scores = f1_score(predicted_labels, ground_truth_labels)
                print("F1:", f1_scores, "Step:", test_step)
                all_predicted_labels.extend(predicted_labels)
                all_ground_truth_labels.extend(ground_truth_labels)

            average_f1_scores = f1_score(all_predicted_labels, all_ground_truth_labels)
            average_accuracy_scores = accuracy_score(all_predicted_labels, all_ground_truth_labels)
            average_precision_scores = precision_score(all_predicted_labels, all_ground_truth_labels)
            average_recall_scores = recall_score(all_predicted_labels, all_ground_truth_labels)

            print("Test with F1 score ", average_f1_scores)
            print("Test with accuracy score ", average_accuracy_scores)
            print("Test with precision score ", average_precision_scores)
            print("Test with recall score ", average_recall_scores)


if __name__ == "__main__":
    main(opt)