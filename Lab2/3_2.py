#!/usr/bin/env python  
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence 
# @file: 3_2.py 
# @time: 2021/02/09
# @contact: mingj@kth,se
# @software: PyCharm 
# May the force be with you.
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

    def delta_train(self, x_train, y_train, x_test, y_test, eta=0.01, pattern="sin", epoch=10, mode='seq', plot=True):
        w = np.random.uniform(-1.0, 1.0, self.num_node)
        n = x_train.shape[0]
        err_list = []
        # y_train = y_train.reshape((y_train.shape[0], ))
        phi_test = self.cal_phi(x_test)
        if mode == 'seq':
            for i in range(n):
                phi_x = self.cal_phi(x_train[i:i + 1])
                err = y_train[i] - np.dot(phi_x, w)
                dw = eta * np.dot(err, phi_x)
                w += dw
                if pattern == 'sin':
                    err_list.append(abs_residual_error(y_train, np.dot(phi_x, w)))
                else:
                    err_list.append(abs_residual_error(y_train, np.dot(phi_x, w)))
        else:
            phi_x = self.cal_phi(x_train)
            for i in range(epoch):
                tmp = np.dot(phi_x, w)
                err = y_train - tmp
                dw = eta * np.dot(err, phi_x)
                w += dw
                if pattern == 'sin':
                    err_list.append(abs_residual_error(y_train, tmp))
                else:
                    err_list.append(abs_residual_error(y_train, tmp))
                # print(error_compute(y_train, tmp))
        y_pred = np.dot(phi_test, w)
        if pattern != "sin":
            y_pred = np.where(y_pred >= 0, 1, -1)
            mse = abs_residual_error(y_test, y_pred)
        else:
            mse = abs_residual_error(y_test, y_pred)
        if plot:
            plt.plot(y_test, label="Truth")
            plt.plot(y_pred, label="Predictions")
            plt.legend()
            plt.title(f"Fitting curve of Pattern: {pattern}, Mode: {mode},Err: {round(mse, 4)}")
            plt.show()
        return y_pred, mse, err_list


def eta_grid_search(eta_list, x_train, y_train, x_test, y_test, pattern='square', epoch=1000, mode='batch'):
    error_list = []
    pred_dict = {}
    # err_dict = {}
    model = RBF(20)
    for i in eta_list:
        y_pred, err = model.lsr_train(x_train, y_train, x_test, y_test, pattern="sin")
        print(f"eta={i}, error={round(err, 4)}.")
        error_list.append(f"eta={i} & {err} \\")

        pred_dict[f"eta={i}"] = y_pred
        print(f"eta={i}, error={err}")
        # err_dict[f"eta={i}"] = err_list
    return error_list, pred_dict


if __name__ == '__main__':
    data = Data()
    noise = True
    x_train, y_sin_train, y_square_train = data.data_task1(noise=noise)
    x_test, y_sin_test, y_square_test = data.data_task1(start=0.05, noise=noise)

    # check eta
    eta_list = [0.01, 0.001, 0.0001]
    # eta_grid_search(eta_list, x_train, y_sin_train, x_test, y_sin_test, pattern='square', epoch=1000, mode='batch')
    # eta_grid_search(eta_list, x_train, y_square_train, x_test, y_square_test, pattern='square', epoch=1000, mode='batch')

    batch_err_list, batch_pred_list = eta_grid_search(eta_list, x_train, y_sin_train, x_test,
                                                                      y_sin_test, pattern='sin', mode='batch')
    # for key, item in batch_err_dict.items():
    #     plt.plot(item, label=key)
    # plt.legend()
    # plt.title("Batch Mode Check for Converge")
    # plt.show()

    batch_err_list, batch_pred_list = eta_grid_search(eta_list, x_train, y_sin_train, x_test,
                                                                      y_sin_test, pattern='sin', mode='seq')
    # for key, item in batch_err_dict.items():
    #     plt.plot(item, label=key)
    # plt.legend()
    # plt.title("On-line Mode Check for Converge")
    # plt.show()