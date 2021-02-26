#!/usr/bin/env python  
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence 
# @file: HopfieldNN.py 
# @time: 2021/02/14
# @contact: mingj@kth,se
# @software: PyCharm 
# May the Force be with you.

from DataGen import *
import matplotlib.pyplot as plt
import itertools


class HopfieldNetwork(object):
    def __init__(self, bias=None):
        self.weight = None
        self.bias = bias
        self.un_updated = None

    # calculate the weight
    def train(self, x, mu=None, diagonal=None, symmetric=False):
        if not mu:
            weight = np.sum(np.outer(p.T, p) for p in x)
        else:
            weight = np.sum(np.outer(p.T - mu, p - mu) for p in x)
        weight = weight.astype(np.float64).copy()

        if diagonal == "diag0":
            np.fill_diagonal(weight, 0)

        if symmetric:
            weight = 0.5 * (weight + weight.T)
        weight /= weight.shape[0]
        # normalization
        self.weight = weight
        return weight

    def energy(self, state):
        return np.sum(-1 * np.dot(np.dot(state.T, self.weight), state))

    """
    :param x_i: previous pattern
    :param x_j: updated pattern
    """

    def update(self, x, mode="synchronous", num=1):
        w = self.weight
        x_i = np.copy(x)
        if mode == "synchronous":
            if self.bias is None:
                # the Little Model
                x_j = np.sign(np.dot(x_i, w))
                x_j[x_j == 0] = 1
            else:
                x_j = np.sign(np.dot(x_i, w) - self.bias)
                x_j[x_j == 0] = 1
                x_j = 0.5 + 0.5 * x_j
            # x_j = np.sign(np.dot(w, x_i))
        else:
            if not self.un_updated:
                self.un_updated = [i for i in range(x.shape[0])]
            update_index = np.random.choice(self.un_updated, num)

            x_j = np.copy(x)
            # print(x.shape, w.shape)
            if self.bias is None:
                x_i = np.sign(np.sum(x * w[update_index, :]))
                x_j[update_index] = x_i
                x_j[x_j == 0] = 1
            else:
                x_i = np.sign(np.sum(x * w[update_index, :]) - self.bias)
                x_j[update_index] = 0.5 + 0.5 * (1 if x_i >= 0 else -1)
            self.un_updated.remove(update_index)
        return x_j

    def recall(self, x, mode="synchronous", iter=100):
        x_i = np.copy(x)
        x_j = np.copy(x)
        for i in range(iter):
            if mode == "synchronous":
                x_j = self.update(x_i, mode=mode)
            else:
                for _ in range(x.shape[0]):
                    x_j = self.update(x_j, mode=mode)
            # check if converge
            # if np.array_equal(x_j, x_i):
            #     break

            # update
            x_i = x_j
        return x_j

    def attractors_count(self, mode="synchronous"):
        nneurons = self.weight.shape[0]
        all_patterns = np.array(list(itertools.product([-1, 1], repeat=nneurons)))
        attractors = []
        for pattern in all_patterns:
            attractors.append(self.recall(pattern, mode=mode))
        attractors = np.unique(np.array(attractors), axis=0)
        return attractors

    def check_memory(self, x, mode="synchronous"):
        count = 0
        memorized = []
        for pattern in x:
            recalled = self.recall(pattern, mode=mode)
            if np.array_equal(recalled, pattern):
                count += 1
                memorized.append(pattern)
        print(f"{count} patterns are memorized")
        return count, memorized


def check_res(model, X, distorted_X, mode="synchronous", iter=100):
    recalled_X = model.recall(distorted_X, mode=mode, iter=iter)
    check = []
    for i in range(X.shape[0]):
        check.append(np.array_equal(recalled_X[i], X[i]))
    print(check)
    return check


if __name__ == '__main__':
    data = Data()
    X, distorted_X, more_error_X = data.get_data(), data.get_data("distorted"), data.get_data("more errors")

    mode = "synchronous"
    iter = 100
    hn = HopfieldNetwork()
    hn.train(X)

    check_res(hn, X, distorted_X, mode=mode, iter=iter)
    check_res(hn, X, more_error_X, mode=mode, iter=iter)

    # ---------------------------------------------------------------------------------------#
    # 3.1 find the attractors
    attractors = hn.attractors_count()
    print(f"{len(attractors)} attractors are found.")
    for attractor in attractors:
        orthogonal = -attractor
        if orthogonal in attractors:
            # print("\hline")
            print(f"{attractor} & {orthogonal} \\\\")
        else:
            print("no")
        # print(np.array(attractor))

    # ---------------------------------------------------------------------------------------#
