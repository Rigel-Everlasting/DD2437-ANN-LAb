#!/usr/bin/env python  
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence 
# @file: CompetitiveLearning.py 
# @time: 2021/02/05
# @contact: mingj@kth,se
# @software: PyCharm 
# May the force be with you.

from RBF_Weight_Learning import *
import random


def cl_learning(x_train, y_train, x_test, y_test, eta, num_node, iter=10000, deadnode=True, plot=False):
    # rbf_nodes = x_train[random.sample(range(0, x_train.shape[0]), num_node)]
    rbf_nodes = x_train[np.arange(num_node)]
    int_list = []
    plt.scatter(x_train, y_train, label="original data")
    plt.scatter(rbf_nodes, np.sin(2 * rbf_nodes), label="initial RBF nodes")
    for i in range(iter):
        train_vector = x_train[np.random.randint(0, len(x_train))]
        int_list.append(int(np.random.randint(0, len(x_train))))
        distances = []
        for node in rbf_nodes:
            distances.append(np.linalg.norm(node - train_vector))
        index = np.argpartition(distances, 1)
        winner = np.argmin(index)
        rbf_nodes[winner] += eta * (train_vector - rbf_nodes[winner])
        if deadnode:
            # Update the worst node (largest distance) to avoid dead nodes
            loser = np.argmax(index)
            rbf_nodes[loser] += eta * eta * (train_vector - rbf_nodes[loser])
    rbf = RBF(num_node)
    rbf.mu = rbf_nodes
    plt.scatter(rbf_nodes, np.sin(2 * rbf_nodes), label="updated RBF nodes", color="red")
    plt.legend()
    plt.show()
    rbf.sigma = 1.0
    pred, err = rbf.lsr_train(x_train, y_train, x_test, y_test, pattern="sin", plot=plot)
    # pred, err = rbf.delta_train(x_train, y_train, x_test, y_test, eta=eta, pattern="sin", epoch=1000, mode='batch',
    #                          plot=plot)
    print(err)
    return pred, err


if __name__ == '__main__':
    data = Data()
    noise = True
    plot = True
    
    x_train, y_sin_train, y_square_train = data.data_task1(noise=noise)
    x_test, y_sin_test, y_square_test = data.data_task1(noise=noise, start=0.05)
    cl_learning(x_train, y_sin_train, x_test, y_sin_test, 0.1, 16, plot=plot)

    model = RBF(16)
    model.sigma = 1.0
    _, err = model.lsr_train(x_train, y_sin_train, x_test, y_sin_test, pattern="sin", plot=plot)
    print(err)

    # ---------------------------------------------------------------------------------#

    # x_train, y_train, x_test, y_test = data.load_data()
    # y_train_dist = y_train[:, :1]
    # y_train_dist = y_train_dist.reshape((y_train_dist.shape[0],))
    #
    # y_train_height = y_train[:, 1:]
    # y_train_height = y_train_height.reshape((y_train_height.shape[0],))
    #
    # y_test_dist = y_test[:, :1]
    # y_test_dist = y_test_dist.reshape((y_test_dist.shape[0],))
    #
    # y_test_height = y_test[:, 1:]
    # y_test_height = y_test_height.reshape((y_test_height.shape[0],))
    #
    # pred_dist, err_dist = cl_learning(x_train, y_train_dist, x_test, y_test_dist, 0.01, 12)
    # pred_height, err_height = cl_learning(x_train, y_train_height, x_test, y_test_height, 0.01, 12)
    #
    # plt.scatter(y_test_dist, y_test_height)
    # plt.scatter(pred_dist, pred_height)
    # plt.xlabel = ("distance")
    # plt.ylabel("height")
    # plt.title("Ballist trained in batch mode, RBF nodes from competitive learning")
    # # plt.title("Ballist trained in batch mode, RBF nodes randomly chosen")
    # plt.show()
