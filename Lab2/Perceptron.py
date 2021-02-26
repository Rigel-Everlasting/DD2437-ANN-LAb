#!/usr/bin/env python  
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence 
# @file: Perceptron.py 
# @time: 2021/02/09
# @contact: mingj@kth,se
# @software: PyCharm 
# May the Force be with you.

from DataGen import *
from RBF_Weight_Learning import *


def error_compute(T, y):
    return np.mean(np.square(T - y))


def misclassification_ratio(T, y):
    predictions = np.where(y >= 0, 1, -1)
    mis = np.where(T != predictions)[0]
    ratio = len(mis) / T.shape[0]
    return ratio


class MLP:
    def __init__(self, learning_rate=0.01, alpha=0.9, epochs=20, Nhidden=5):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.Nhidden = Nhidden
        self.w_list = []
        self.v_list = []

    def phi(self, x):
        output = 2 / (1 + np.exp(-x)) - 1
        return output

    def derivative(self, x):
        return 0.5 * (1 + self.phi(x)) * (1 - self.phi(x))

    def forward_pass(self, X, W, V):
        H_in = np.dot(W, X)
        H_out = self.phi(H_in)
        # add a bias term, an input signal in each layer with the value 1
        H_out = np.vstack([H_out, np.ones(np.shape(H_out)[1])])
        O_in = np.dot(V, H_out)
        O_out = self.phi(O_in)
        return H_out, O_out

    def backward_pass(self, T, H, O, V):
        delta_o = (O - T) * self.derivative(O)
        delta_h = np.dot(V.T, delta_o) * self.derivative(H)
        # remove the bias term
        delta_h = delta_h[0:self.Nhidden]
        return delta_h, delta_o

    def weight_update(self, X, H, W, V, dw, dv, delta_h, delta_o):
        dw = self.alpha * dw - (1 - self.alpha) * np.dot(delta_h, X.T)
        dv = self.alpha * dv - (1 - self.alpha) * np.dot(delta_o, H.T)
        W = W + self.learning_rate * dw
        V = V + self.learning_rate * dv
        return W, V, dw, dv

    def back_propagation(self, X, T, W, V, binary=True):
        errors = []
        misclassification_ratios = []
        self.w_list = []
        self.v_list = []
        dw = np.ones(np.shape(W))
        dv = np.ones(np.shape(V))

        for i in range(self.epochs):
            H_out, O_out = self.forward_pass(X, W, V)
            errors.append(error_compute(T, O_out))
            misclassification_ratios.append(misclassification_ratio(T, O_out))
            delta_h, delta_o = self.backward_pass(T, H_out, O_out, V)

            # update
            W, V, dw, dv = self.weight_update(X, H_out, W, V, dw, dv,
                                              delta_h, delta_o)
            self.w_list.append(W)
            self.v_list.append(V)

        _, predictions = self.forward_pass(X, W, V)
        if binary:
            predictions = np.where(predictions >= 0, 1, -1)
        return W, V, predictions, errors, misclassification_ratios

    def predict(self, X, W, V, binary=True):
        _, predictions = self.forward_pass(X, W, V)
        if binary:
            predictions = np.where(predictions >= 0, 1, -1)
        return predictions


if __name__ == '__main__':
    data = Data()
    noise = True
    epochs = 300
    nnodes_list = [8, 12, 20]
    x_train, y_sin_train, y_square_train = data.data_task1(noise=noise)
    x_test, y_sin_test, y_square_test = data.data_task1(start=0.05, noise=noise)
    x_train_ = x_train
    x_test_ = x_test

    # avoid dimension error in perceptron learning
    x_train = x_train.reshape((x_train.shape[0], 1)).T
    x_test = x_test.reshape((x_test.shape[0], 1)).T

    # ---------------------------------------------------------------------------------#

    # sin(2x)
    print(f"Nodes Numbers & RBF MSE & Perceptron MSE " + r"\\")
    per_sin_errs = []
    rbf_sin_errs = []
    for nnodes in nnodes_list:
        Nhidden =nnodes
        mlp = MLP(epochs=epochs, Nhidden=Nhidden)
        init_w1 = np.random.randn(Nhidden, np.shape(x_train)[0])
        init_w2 = np.random.randn(1, Nhidden + 1)
        w1, w2, pred, err, _ = mlp.back_propagation(x_train, y_sin_train, init_w1, init_w2)
        y_sin_per = mlp.predict(x_test, w1, w2, binary=False)
        per_sin_err = error_compute(y_sin_test, y_sin_per)
        per_sin_errs.append(per_sin_err)

        rbf = RBF(nnodes)
        y_sin_rbf, err = rbf.delta_train(x_train_, y_sin_train, x_test_, y_sin_test, epoch=epochs, mode='batch', plot=False)
        rbf_sin_err = error_compute(y_sin_test, y_sin_rbf)
        rbf_sin_errs.append(rbf_sin_err)

        print(f"{nnodes} & {round(rbf_sin_err, 4)} & {round(per_sin_err, 4)} " + r"\\")

    # ---------------------------------------------------------------------------------#

    # square
    print(f"Nodes Numbers & RBF MSE & Perceptron MSE " + r"\\")
    per_sqr_errs = []
    rbf_sqr_errs = []
    for nnodes in nnodes_list:
        Nhidden = nnodes
        mlp = MLP(epochs=epochs, Nhidden=Nhidden)
        init_w1 = np.random.randn(Nhidden, np.shape(x_train)[0])
        init_w2 = np.random.randn(1, Nhidden + 1)
        w1, w2, pred, err, _ = mlp.back_propagation(x_train, y_square_train, init_w1, init_w2)
        y_sqr_per = mlp.predict(x_test, w1, w2, binary=True)
        per_sqr_err = error_compute(y_square_test, y_sqr_per)
        per_sqr_errs.append(per_sqr_err)

        rbf = RBF(nnodes)
        y_sqr_rbf, err = rbf.delta_train(x_train_, y_square_train, x_test_, y_sin_test, pattern="square",
                                         epoch=epochs, mode='batch', plot=False)
        rbf_sqr_err = error_compute(y_square_test, y_sqr_rbf)
        rbf_sqr_errs.append(rbf_sqr_err)

        print(f"{nnodes} & {round(rbf_sqr_err, 4)} & {round(per_sqr_err, 4)} " + r"\\")

