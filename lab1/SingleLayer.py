#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
# @author: JamieJ
# @license: Apache Licence 
# @file: SingleLayer.py 
# @time: 2021/01/23
# @contact: mingj@kth,se
# @software: PyCharm
# May the Force be with you.
from DataGen import *


def process_input(classA, classB, bias=True):
    _, n = np.shape(classA)
    if bias:
        classA = np.concatenate((classA, np.ones((1, n))))
        classB = np.concatenate((classB, np.ones((1, n))))

    ###Get Label
    classA = np.concatenate((classA, np.ones((1, n))))
    classB = np.concatenate((classB, -np.ones((1, n))))
    X = np.concatenate((classA, classB), axis=1)
    X = np.transpose(X)
    np.random.shuffle(X)
    X = np.transpose(X)
    W = np.random.randn(1, 3)
    return X[0:3, :], X[3, :], W

def delta_rule(X, T, init_w, learning_rate, epochs=20):
    W = init_w.copy()
    errors = []
    for i in range(epochs):
        y = np.dot(W, X)
        error = T - y
        dw = learning_rate * error.dot(X.T)
        W += dw
        errors.append(np.mean(np.square(error)))
    return W, errors

def perceptron_learning(X, T, init_w, learning_rate, epochs=20):
    W = init_w.copy()
    errors = []
    for i in range(epochs):
        y = np.dot(W, X)
        y = np.where(y >= 0, 1, -1)
        error = T - y
        dw = learning_rate * error.dot(X.T)
        W += dw
        errors.append(np.mean(np.square(error)))
    return W, errors


def sequential_delta_rule(X, T, W,eta,epochs):
  errors_list=[]
  for i in range(epochs):
    for k in range(np.shape(T)[0]):
      tmp=[]
      error=W.dot(X[:,k])-T[k]
      tmp.append(error**2)
      W=W-eta*error*X[:,k].T
    errors_list.append(np.mean(tmp))
  return W,errors_list

def sequential_delta(X, T, W, learning_rate, epochs=20):
    errors = []
    for k in range(epochs):
        for i in range(T.shape[0]):
            mean_error = []
            y = np.dot(W, X[:, i])
            e = T[i] - y
            W -= learning_rate * e * X[:, i].T
            mean_error.append(e**2)
        errors.append(np.mean(mean_error))
    return W, errors

def plot_boundry(W, classA, classB):
    w1 = W[0][0]
    w2 = W[0][1]
    w3 = W[0][2]
    x = np.linspace(-2,2,1000)
    y = (-1/w2)*(w1*x+w3)
    plt.plot(classA[0],classA[1],'go')
    plt.plot(classB[0],classB[1],'bo')
    plt.plot(x,y,'r-')
    plt.legend(['class A','class B'])
    plt.show()

if __name__ == '__main__':
    data = DataGenerator()
    classA, classB = data.lin_data()
    X, T, Weight = process_input(classA, classB)
    learning_rate = 0.001
    epochs = 30
    # delta_w, delta_e = delta_rule(X, T, Weight, learning_rate, epochs)
    # per_w, per_e = perceptron_learning(X, T, Weight, learning_rate, epochs)
    # sdelta_w, sdelta_e = sequential_delta_rule(X, T, Weight, learning_rate, epochs)
    plot_boundry(Weight, classA, classB)
    # plot_boundry(delta_w, classA, classB)
    # plot_boundry(per_w, classA, classB)
    error_list = []
    learning_rate_list = [0.0001, 0.0005, 0.001]
    for l in learning_rate_list:
        d_w, d_e = delta_rule(X, T, Weight, l, epochs)
        plot_boundry(d_w, classA, classB)
        error_list.append(d_e)
        # plt.plot(d_e, label=f"learning rate={l}")
        # plt.show()

    for i in range(len(error_list)):
        plt.plot(error_list[i], label=f"learning rate={learning_rate_list[i]}")
    # plt.plot(per_e, label="perceptron learning")
    plt.xlabel = ("epochs")
    plt.ylabel("errors")
    plt.legend()
    plt.show()