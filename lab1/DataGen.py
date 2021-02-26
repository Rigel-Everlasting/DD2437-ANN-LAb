#!/usr/bin/env python  
# -*- coding:utf-8 _*-
# @author:Clarky Clark Wang
# @license: Apache Licence 
# @file: DataGen.py 
# @time: 2021/01/23
# @contact: wangz@kth,se
# @software: PyCharm 
# Import Libs and Let's get started, shall we?
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self):
        self.lower = 301
        self.higher = 1501
        self.count = 1200
        self.predict = 5

    def lin_data(self, n=100, mA=np.array([2, 1.5]), mB=np.array([-1.5, 0]), sigmaA=0.5, sigmaB=0.5):
        classA = np.zeros((2, n))
        classB = np.zeros((2, n))

        for i in range(2):
            classA[i, :] = np.random.normal(mA[i], sigmaA, n)
            classB[i, :] = np.random.normal(mB[i], sigmaB, n)

        return classA, classB

    def non_lin_data(self, n=100, mA=np.array([1.0, 0.3]), mB=np.array([0.0, -0.1]), sigmaA=0.2, sigmaB=0.3):
        classA = np.zeros((2, n))
        classB = np.zeros((2, n))

        for i in range(2):
            if i == 0:
                classA[i, 0:n // 2] = np.random.normal(-mA[i], sigmaA, n // 2)
                classA[i, n // 2:] = np.random.normal(mA[i], sigmaA, n // 2)
            else:
                classA[i, :] = np.random.normal(mA[i], sigmaA, n)
            classB[i, :] = np.random.normal(mB[i], sigmaB, n)

        return classA, classB

    def plot_classes(self, classA, classB, type=True):
        ax = plt.subplot(111)
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        plt.axis('equal')
        if type:
            plt.title('Linearly-separable data')
        else:
            plt.title('Non-linearly-separable data')
        plt.scatter(classA[0, :], classA[1, :], color='red')
        plt.scatter(classB[0, :], classB[1, :], color='blue')
        plt.legend(['A', 'B'])
        plt.show()

    ### Part Two
    ### Generate Time Series Data
    def mackey_glass(self, beta=0.2, gamma=0.1, n=10, tao=25):
        iter = self.higher + self.predict
        x = np.zeros(iter + 1)
        x[0] = 1.5
        for t in range(iter):
            res = t - tao
            if res < 0:
                res = 0
            elif res == 0:
                res = x[0]
            else:
                res = x[res]
            x[t + 1] = x[t] + (beta * res) / (1 + res ** n) - gamma * x[t]
        return x

    def time_series(self, x, sd=0.09, points=1200):
        t = np.arange(self.lower, self.higher, 1)
        input = np.array([x[t - 20], x[t - 15], x[t - 10], x[t - 5], x[t]])
        output = x[t + self.predict] + np.random.normal(0, sd, points)
        return input.T, output


if __name__ == '__main__':
    data = DataGenerator()
