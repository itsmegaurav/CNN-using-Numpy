import numpy as np
from sklearn.utils import shuffle
from deepnet.utils import accuracy
import copy
from deepnet.loss import SoftmaxLoss


def get_minibatches(X, y, minibatch_size,shuffle=True):
    m = X.shape[0]
    minibatches = []
    if shuffle:
        X, y = shuffle(X, y)
    for i in range(0, m, minibatch_size):
        X_batch = X[i:i + minibatch_size, :, :, :]
        y_batch = y[i:i + minibatch_size, ]
        minibatches.append((X_batch, y_batch))
    return minibatches


def vanilla_update(params, grads, learning_rate=0.01):
    for param, grad in zip(params, reversed(grads)):
        for i in range(len(grad)):
            param[i] += - learning_rate * grad[i]


def sgd(nnet, X_train, y_train, minibatch_size, epoch, learning_rate, verbose=True,
        X_test=None, y_test=None):
    minibatches = get_minibatches(X_train, y_train, minibatch_size)
    for i in range(epoch):
        loss = 0
        if verbose:
            print("Epoch {0}".format(i + 1))
        for X_mini, y_mini in minibatches:
            loss, grads = nnet.train_step(X_mini, y_mini)
            vanilla_update(nnet.params, grads, learning_rate=learning_rate)
        if verbose:
            train_acc = accuracy(y_train, nnet.predict(X_train))
            test_acc = accuracy(y_test, nnet.predict(X_test))
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".format(
                loss, train_acc, test_acc))
    return nnet




