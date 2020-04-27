import pandas as pd
import random
import torch
import time
import numpy as np
from gensim.models.word2vec import Word2Vec
from model import BatchProgramClassifier
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        if item[0] ==0:
            labels.append(0)
        elif item[0] ==1:
            labels.append(1)
    return data, torch.LongTensor(labels)


if __name__ == '__main__':
    root = './'
    train_data = pd.read_pickle(root+'imtrain/bal_blocks.pkl')
    val_data = pd.read_pickle(root + 'imdev/bal_blocks.pkl')
    test_data = pd.read_pickle(root+'imtest/blocks1.pkl')

    word2vec = Word2Vec.load(root+"imtrain/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    HIDDEN_DIM = 100
    ENCODE_DIM = 128
    LABELS = 2
    EPOCHS = 100
    BATCH_SIZE = 64
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS+1,ENCODE_DIM,LABELS,BATCH_SIZE,
                                   USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.CrossEntropyLoss()

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    train_recall_ = []
    val_recall_ = []
    train_precision_ = []
    val_precision_ = []
    train_MCC_ = []
    val_MCC_ = []
    train_TP_ = 0
    train_FN_ = 0
    train_FP_ = 0
    train_TN_ = 0
    val_TP_ = 0
    val_FN_ = 0
    val_FP_ = 0
    val_TN_ = 0
    best_acc = 0.0
    print('Start training...')
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total_recall = 0.0
        total_precision = 0.0
        total_MCC = 0.00
        total_TP = 0.0
        total_FN = 0.0
        total_FP = 0.0
        total_TN = 0.0
        total = 0.0
        i = 0
        while i < len(train_data):
            batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = batch
            if USE_GPU:
                train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs)

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total_TP += ((predicted == train_labels)*(predicted > 0)).sum()
            total_FN += (predicted < train_labels).sum()
            total_FP += (predicted > train_labels).sum()
            total_TN += ((predicted == train_labels)*(predicted == 0)).sum()
            if total_TP.item()+total_FN.item()>0:
                total_recall = total_TP.item() / (total_TP.item()+total_FN.item())
            if total_TP.item()+total_FP.item()>0:
                total_precision = total_TP.item() / (total_TP.item()+total_FP.item())
            if (total_TP.item()+total_FP.item())*(total_TP.item()+total_FN.item())*(total_TN.item()+total_FP.item())*(total_TN.item()+total_FN.item())>0:
                M = (total_TP.item()+total_FP.item())*(total_TP.item()+total_FN.item())*(total_TN.item()+total_FP.item())*(total_TN.item()+total_FN.item())
                total_MCC = (total_TP.item()*total_TN.item()-total_FP.item()*total_FN.item())/np.sqrt(M)
            total += len(train_labels)
            total_loss += loss.item()*len(train_inputs)

        train_TP_ = total_TP
        train_FN_ = total_FN
        train_FP_ = total_FP
        train_TN_ = total_TN
        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
        train_recall_.append(total_recall)
        train_precision_.append(total_precision)
        train_MCC_.append(total_MCC)

        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total_recall = 0.0
        total_precision = 0.0
        total_MCC = 0.00
        total_TP = 0.0
        total_FN = 0.0
        total_FP = 0.0
        total_TN = 0.0
        total = 0.0
        i = 0
        while i < len(val_data):
            batch = get_batch(val_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            val_inputs, val_labels = batch
            if USE_GPU:
                val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output = model(val_inputs)

            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total_TP += ((predicted == val_labels)*(predicted > 0)).sum()
            total_FN += (predicted < val_labels).sum()
            total_FP += (predicted > val_labels).sum()
            total_TN += ((predicted == val_labels)*(predicted == 0)).sum()
            if total_TP.item()+total_FN.item()>0:
                total_recall = total_TP.item() / (total_TP.item()+total_FN.item())
            if total_TP.item()+total_FP.item()>0:
                total_precision = total_TP.item() / (total_TP.item()+total_FP.item())
            if (total_TP.item()+total_FP.item())*(total_TP.item()+total_FN.item())*(total_TN.item()+total_FP.item())*(total_TN.item()+total_FN.item())>0:
                M = (total_TP.item()+total_FP.item())*(total_TP.item()+total_FN.item())*(total_TN.item()+total_FP.item())*(total_TN.item()+total_FN.item())
                total_MCC = (total_TP.item()*total_TN.item()-total_FP.item()*total_FN.item())/np.sqrt(M)
            total += len(val_labels)
            total_loss += loss.item()*len(val_inputs)
        
        val_TP_ = total_TP
        val_FN_ = total_FN
        val_FP_ = total_FP
        val_TN_ = total_TN
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        val_recall_.append(total_recall)
        val_precision_.append(total_precision)
        val_MCC_.append(total_MCC)

        end_time = time.time()
        if total_acc/total > best_acc:
            best_model = model
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
              ' Training Acc: %.3f, Validation Acc: %.3f, Training Recall: %.3f,Validation Recall: %.3f,Training Precision: %.3f,Validation Precision: %.3f,Training MCC: %.3f,Validation MCC: %.3f,Time Cost: %.3f s'
              % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch],train_recall_[epoch],val_recall_[epoch],train_precision_[epoch],val_precision_[epoch],train_MCC_[epoch],val_MCC_[epoch],end_time - start_time))

    torch.save(model.state_dict(), "./saved_model/balanced_imtrainfolder_net.pkl")
    
    total_acc = 0.0
    total_loss = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_MCC = 0.00
    total_TP = 0.0
    total_FN = 0.0
    total_FP = 0.0
    total_TN = 0.0
    total = 0.0
    i = 0
    model = best_model
    while i < len(test_data):
        batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()

        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == test_labels).sum()
        total_TP += ((predicted == test_labels)*(predicted > 0)).sum()
        total_FN += (predicted < test_labels).sum()
        total_FP += (predicted > test_labels).sum()
        total_TN += ((predicted == test_labels)*(predicted == 0)).sum()
        if total_TP.item()+total_FN.item()>0:
            total_recall = total_TP.item() / (total_TP.item()+total_FN.item())
        if total_TP.item()+total_FP.item()>0:
            total_precision = total_TP.item() / (total_TP.item()+total_FP.item())
        if (total_TP.item()+total_FP.item())*(total_TP.item()+total_FN.item())*(total_TN.item()+total_FP.item())*(total_TN.item()+total_FN.item())>0:
            M = (total_TP.item()+total_FP.item())*(total_TP.item()+total_FN.item())*(total_TN.item()+total_FP.item())*(total_TN.item()+total_FN.item())
            total_MCC = (total_TP.item()*total_TN.item()-total_FP.item()*total_FN.item())/np.sqrt(M)
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print("Testing results(Acc):", total_acc.item() / total)
    print("Testing Recall):", total_recall)
    print("Testing Precision:", total_precision)
    print("Testing MCC:", total_MCC)
    print("Testing TP:", total_TP)
    print("Testing FN:", total_FN)
    print("Testing FP:", total_FP)
    print("Testing TN:", total_TN)