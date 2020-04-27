import numpy as np

import tensorflow as tf


def cul_f1(y_hat, y_true, model='multi'):
    '''
    输入张量y_hat是输出层经过sigmoid激活，四舍五入之后的张量
    '''
    epsilon = 1e-7
    y_hat = np.asarray(y_hat, np.float32)
    y_true = np.asarray(y_true, np.float32)
    y_hat = tf.convert_to_tensor(y_hat, np.float32)
    y_true = tf.convert_to_tensor(y_true,np.float32)
    tp = tf.reduce_sum(tf.cast(y_hat * y_true, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_hat) * y_true, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_hat * (1 - y_true), 'float'), axis=0)

    p = tp / (tp + fp + epsilon)  # epsilon的意义在于防止分母为0，否则当分母为0时python会报错
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    if model == 'single':
        return f1
    if model == 'multi':
        return tf.reduce_mean(f1)

def cul_recall(y_hat, y_true):
    '''
    输入张量y_hat是输出层经过sigmoid激活，四舍五入之后的张量
    '''
    epsilon = 1e-7
    y_hat = np.asarray(y_hat, np.float32)
    y_true = np.asarray(y_true, np.float32)
    y_hat = tf.convert_to_tensor(y_hat, np.float32)
    y_true = tf.convert_to_tensor(y_true, np.float32)
    tp = tf.reduce_sum(tf.cast(y_hat * y_true, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_hat * (1 - y_true), 'float'), axis=0)

    r = tp / (tp + fn + epsilon)

    return r

def cul_precision(y_hat, y_true):
    '''
    输入张量y_hat是输出层经过sigmoid激活，四舍五入之后的张量
    '''
    epsilon = 1e-7
    y_hat = np.asarray(y_hat, np.float32)
    y_true = np.asarray(y_true, np.float32)
    y_hat = tf.convert_to_tensor(y_hat, np.float32)
    y_true = tf.convert_to_tensor(y_true, np.float32)
    tp = tf.reduce_sum(tf.cast(y_hat * y_true, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_hat) * y_true, 'float'), axis=0)

    p = tp / (tp + fp + epsilon)

    return p

def cul_acc(y_hat, y_true):
    '''
    输入张量y_hat是输出层经过sigmoid激活,四舍五入之后的张量
    '''
    y_hat = np.asarray(y_hat, np.float32)
    y_true = np.asarray(y_true, np.float32)
    y_hat = tf.convert_to_tensor(y_hat, np.float32)
    y_true = tf.convert_to_tensor(y_true, np.float32)
    tp = tf.reduce_sum(tf.cast(y_hat * y_true, 'float'), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_hat) * y_true, 'float'), axis=0)
    fn = tf.reduce_sum(tf.cast(y_hat * (1 - y_true), 'float'), axis=0)
    tn = tf.reduce_sum(tf.cast((1 - y_hat) * (1-y_true), 'float'), axis=0)

    a = (tp+tn) / (tp + tn + fp + fn)

    return a


def calculate_f1_scores(predictions, ground_truths):
    f1_scores = []
    for i, prediction in enumerate(predictions):
        f1_score = cul_f1(prediction, ground_truths[i])
        f1_scores.append(f1_score)

    return tf.reduce_mean(f1_scores)