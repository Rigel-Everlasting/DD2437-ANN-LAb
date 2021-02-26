#!/usr/bin/env python  
# -*- coding:utf-8 _*-
# @author:Clarky Clark Wang
# @license: Apache Licence 
# @file: DataGen.py 
# @time: 2021/02/03
# @contact: wangz@kth,se
# @software: PyCharm 
# Import Libs and Let's get started, shall we?
import numpy as np
import matplotlib.pyplot as plt


class Data(object):
    def data_task1(self, start=0.0, end=2 * np.pi, step_size=0.1, noise=False):
        x = np.arange(start, end, step_size)
        sin_y = np.sin(2 * x)
        if noise:
            sin_y += np.random.normal(0, 0.1, x.shape)
        else:
            pass
        square_y = np.where(sin_y >= 0, 1, -1)
        return x, sin_y, square_y

    def load_data(self):
        train = np.loadtxt("data_lab2/ballist.dat")
        test = np.loadtxt("data_lab2/balltest.dat")
        x_train, y_train = train[:, :2], train[:, 2:]
        x_test, y_test = test[:, :2], test[:, 2:]
        return x_train, y_train, x_test, y_test

    def plot_data_task1(self, noise=False):
        x, sin_y, square_y = self.data_task1(noise = noise)
        plt.plot(x, sin_y, label="sin(2x)")
        plt.plot(x, square_y, label="square(2x)")
        plt.legend(loc="lower left")
        plt.show()


if __name__ == '__main__':
    data = Data()
    data.plot_data_task1(noise=False)
    data.plot_data_task1(noise=True)
