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


def phi(x, mu, sigma):
    return np.exp(- np.square(np.sum((x - mu) ** 2)) / (2 * sigma ** 2))


def abs_residual_error(y, pred):
    return np.mean(abs(y - pred))


def error_compute(y, y_pred):
    return np.mean(np.square(y - y_pred))


class RBF(object):
    def __init__(self, num_node):
        self.num_node = num_node
        self.mu = [(i + 0.5) * 2 * np.pi / num_node for i in range(num_node)]
        self.sig = 2 * np.pi / num_node
        self.init_weight = np.random.uniform(0, 2 * np.pi, num_node)

    def cal_phi(self, x_train):
        phi_matrix = np.array([[phi(x_train[j], self.mu[i], self.sig)
                                for i in range(self.num_node)] for j in range(len(x_train))])
        return phi_matrix

    def lsr_train(self, x_train, y_train, x_test, y_test, pattern="sin", plot=True):
        phi_x = self.cal_phi(x_train)
        phi_test = self.cal_phi(x_test)
        w = np.linalg.solve(np.dot(phi_x.T, phi_x), np.dot(phi_x.T, y_train))
        y_pred = np.dot(phi_test, w)
        if pattern != "sin":
            y_pred = np.where(y_pred >= 0, 1, -1)
        abs_re = abs_residual_error(y_test, y_pred)
        if plot:
            plt.plot(y_test, label="Truth")
            plt.plot(y_pred, label="Predictions")
            plt.legend()
            plt.title(f"Fitting curve")
            plt.show()
        return y_pred, abs_re

    def delta_train(self, x_train, y_train, x_test, y_test, eta=0.001, pattern="sin", epoch=10, mode='seq', plot=True):
        w = np.random.uniform(-1.0, 1.0, self.num_node)
        n = x_train.shape[0]
        # y_train = y_train.reshape((y_train.shape[0], ))
        phi_test = self.cal_phi(x_test)
        if mode == 'seq':
            for i in range(n):
                phi_x = self.cal_phi(x_train[i:i + 1])
                err = y_train[i] - np.dot(phi_x, w)
                dw = eta * np.dot(err, phi_x)
                w += dw
        else:
            phi_x = self.cal_phi(x_train)
            for i in range(epoch):
                tmp = np.dot(phi_x, w)
                err = y_train - tmp
                dw = eta * np.dot(err, phi_x)
                w += dw
                # print(error_compute(y_train, tmp))
        y_pred = np.dot(phi_test, w)
        if pattern != "sin":
            y_pred = np.where(y_pred >= 0, 1, -1)
            mae = abs_residual_error(y_test, y_pred)
        else:
            mae = abs_residual_error(y_test, y_pred)
        if plot:
            plt.plot(y_test, label="Truth")
            plt.plot(y_pred, label="Predictions")
            plt.legend()
            plt.title(f"Fitting curve of Pattern: {pattern}, Mode: {mode}")
            plt.show()
        return y_pred, mae


def grid_search(nnodes_list, sigma_list, x_train, y_train, x_test, y_test,
                pattern='square', epoch=500, mode='batch', plot=True):
    error_list = []
    print(f"Nodes num & Sigma(nodes width) & Error " + r"\\")
    for nnodes in nnodes_list:
        model = RBF(nnodes)
        for sigma in sigma_list:
            model.sig = sigma
            if mode == 'batch':
                _, err = model.lsr_train(x_train, y_train, x_test, y_test, pattern=pattern, plot=plot)
                print(f"{nnodes} & {sigma} & {round(err, 4)} " + r"\\")
                error_list.append(f"{nnodes} & {sigma} & {round(err, 4)} ")
            else:
                _, err = model.delta_train(x_train, y_train, x_test, y_test, epoch=epoch, mode='batch', pattern=pattern,
                                           plot=plot)
                print(f"{nnodes} & {sigma} & {round(err, 4)} " + r"\\")
                error_list.append(f"{nnodes} & {sigma} & {round(err, 4)} ")
    return error_list


if __name__ == '__main__':
    data = Data()
    noise = False
    x_train, y_sin_train, y_square_train = data.data_task1(noise=noise)
    x_test, y_sin_test, y_square_test = data.data_task1(start=0.05, noise=noise)

    noise = True
    n_x_train, n_y_sin_train, n_y_square_train = data.data_task1(noise=noise)
    n_x_test, n_y_sin_test, n_y_square_test = data.data_task1(start=0.05, noise=noise)

    # model = RBF()
    # _, err = model.delta_train(x_train, y_sin_train, x_test, y_sin_test, epoch=3000, mode='batch')
    # _, err = model.delta_train(x_train, y_square_train, x_test, y_square_test, pattern='sin', epoch=500,
    #                            mode='batch')
    # print(err)

    # ---------------------------------------------------------------------------------#

    # # noise data train, clean data test
    # model = RBF(20)
    # model.sig = 0.2
    # _, err = model.lsr_train(n_x_train, n_y_sin_train, x_test, y_sin_test, pattern="sin", plot=True)
    # print(round(err, 4))
    #
    # # noise data train, noise data test
    # _, err = model.lsr_train(n_x_train, n_y_sin_train, n_x_test, n_y_sin_test, pattern="sin", plot=True)
    # print(round(err, 4))

    # ---------------------------------------------------------------------------------#

    # # error threshold check
    # # output as the latex table format
    # print(f"Threshold & Absolute Residual Error & Hidden Nodes " + r"\\")
    # check01, check001, check0001 = False, False, False
    # for nodes in range(2, 50):
    #     model = RBF(nodes)
    #     _, err = model.lsr_train(x_train, y_sin_train, x_test, y_sin_test, pattern="sin", plot=False)
    #     if err < 0.1 and not check01:
    #         print(f"0.1 & {round(err, 4)} & {nodes} " + r"\\")
    #         check01 = True
    #     if err < 0.01 and not check001:
    #         print(f"0.01 & {round(err, 4)} & {nodes} " + r"\\")
    #         check001 = True
    #     if err < 0.001 and not check0001:
    #         print(f"0.001 & {round(err, 4)} & {nodes} " + r"\\")
    #         check0001 = True

    # ---------------------------------------------------------------------------------#

    # # relation between nnodes, sigma, and error
    # output as the latex table format
    # nnodes_list = [i for i in range(5, 7)]
    nnodes_list = [8, 16, 20]
    sigma_list = [0.2, 0.6, 1.0]
    print("batch mode, sin(2x), clean data")
    _ = grid_search(nnodes_list, sigma_list, x_train, y_sin_train, x_test, y_sin_test, pattern="sin",
                    plot=False)

    print("batch mode, square(2x), clean data")
    _ = grid_search(nnodes_list, sigma_list, x_train, y_square_train, x_test, y_square_test, plot=False)

    print("online mode, sin(2x), clean data")
    _ = grid_search(nnodes_list, sigma_list, x_train, y_sin_train, x_test, y_sin_test, pattern="sin",
                    mode="online", plot=False)

    print("online mode, square(2x), clean data")
    _ = grid_search(nnodes_list, sigma_list, x_train, y_square_train, x_test, y_square_test,
                    mode="online", plot=False)

    # ---------------------------------------------------------------------------------#

    print("batch mode, sin(2x), noise data")
    _ = grid_search(nnodes_list, sigma_list, n_x_train, n_y_sin_train, n_x_test, n_y_sin_test, pattern="sin",
                    plot=False)

    print("batch mode, square(2x), noise data")
    _ = grid_search(nnodes_list, sigma_list, n_x_train, n_y_square_train, n_x_test, n_y_square_test, plot=False)

    print("online mode, sin(2x), noise data")
    _ = grid_search(nnodes_list, sigma_list, n_x_train, n_y_sin_train, n_x_test, n_y_sin_test, mode="online",
                    pattern="sin", plot=False)

    print("online mode, square(2x), noise data")
    _ = grid_search(nnodes_list, sigma_list, n_x_train, n_y_square_train, n_x_test, n_y_square_test,
                    mode="online", plot=False)
