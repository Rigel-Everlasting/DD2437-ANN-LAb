import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def sin_2x(x):
    return np.sin(2 * x)


def square_2x(x):
    y = sin_2x(x)
    y = np.where(y >= 0, 1, -1)
    return y


def data_gen(pattern, start=0, end=2 * np.pi, step_size=0.1, noise=False):
    X = np.arange(start, end, step_size)
    if pattern == "sin":
        y = sin_2x(X)
    elif pattern == "square":
        y = square_2x(X)
    if noise:
        X += np.random.normal(0, np.sqrt(0.1))
    return X, y


# Guassian RBFs
# implement on the hidden layers
def phi(x, mu, sigma):
    return np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def init_nodes(nnodes):
    # mu_list = np.random.uniform(0, 2*np.pi, nnodes)
    mu_list = [(i + 0.5) * 2 * np.pi / nnodes for i in range(nnodes)]
    # sigma = 2 * np.pi / nnodes
    sigma = 0.7
    return sigma, mu_list


def init_weights(nnodes):
    W = np.random.uniform(-1.0, 1.0, nnodes)
    # W = np.random.rand(nnodes, )
    return W


def phi_matrix(nnodes, x_train):
    # init_phi = np.zeros((nnodes, nfeatures))
    sigma, mu = init_nodes(nnodes)
    # phi_x = np.array([[phi(x, mu[i], sigma) for i in range(nnodes)]
    #                   for x in x_train])
    phi_x = np.array([[phi(x_train[j], mu[i], sigma) for i in range(nnodes)]
                      for j in range(len(x_train))])
    return phi_x


def abs_residual_error(y, pred):
    return np.mean(abs(y - pred))


def rbf_ls(x_train, y_train, x_test, y_test, nnodes=5, pattern="sin"):
    phi_x = phi_matrix(nnodes, x_train)
    w = np.linalg.solve(np.dot(phi_x.T, phi_x), np.dot(phi_x.T, y_train))
    phi_test = phi_matrix(nnodes, x_test)
    y_pred = np.dot(phi_test, w)
    if pattern != "sin":
        y_pred = np.where(y_pred >= 0, 1, -1)

    abs_re = abs_residual_error(y_test, y_pred)
    return y_pred, abs_re


def error_compute(y, y_pred):
    return np.mean(np.square(y - y_pred))


# batch mode delta rule
def rbf_delta1(x_train, y_train, x_test, y_test,
               nnodes=5, learning_rate=0.0001, pattern="sin"):
    w = init_weights(nnodes)
    n = x_train.shape[0]
    for i in range(n):
        phi_x = phi_matrix(nnodes, x_train[i:i + 1])
        e = y_train[i] - np.dot(phi_x, w)
        dw = learning_rate * np.dot(e, phi_x)
        w += dw
    phi_test = phi_matrix(nnodes, x_test)
    y_pred = np.dot(phi_test, w)
    mse = error_compute(y_test, y_pred)
    return y_pred, mse


# batch mode delta rule
def rbf_delta(x_train, y_train, x_test, y_test,
              nnodes=5, epochs=500, learning_rate=0.001, pattern="sin"):
    w = init_weights(nnodes)
    phi_x = phi_matrix(nnodes, x_train)
    # k = 0
    for epoch in range(epochs):

        # error = y_train[k] - np.dot(phi_x[k], w)
        # k = (k+1) % x_train.shape[0]
        # dw = learning_rate * error * phi_x[k]
        error = y_train - np.dot(phi_x, w)
        dw = learning_rate * np.dot(error, phi_x)
        # print(dw.shape, error.shape, phi_x.shape, phi_x.T.shape)
        w += dw
    phi_test = phi_matrix(nnodes, x_test)
    y_pred = np.dot(phi_test, w)
    if pattern != "sin":
        y_pred = np.where(y_pred >= 0, 1, -1)
    mse = error_compute(y_test, y_pred)
    return y_pred, mse


def fitting_curve(y, y_pred, title):
    plt.plot(y, label="Truth")
    plt.plot(y_pred, label="Predictions")
    plt.legend()
    plt.title(f"Fitting curve of {title}")


def plot_error(errors, title):
    plt.plot(errors)
    plt.title(f'Learning error curve for the {title}')
    plt.xlabel = ("nnodes")
    plt.ylabel("Error")
    plt.show()


# change by nnodes
def rbf_nnodes(nnodes_list, x_train, y_train, x_test, y_test, pattern="sin", rbf="lse"):
    plt.plot(y_test, label="Truth", lw=3.5)
    error_lists = []
    if rbf == "lse":
        for nnodes in nnodes_list:
            pred, error = rbf_ls(x_train, y_train, x_test, y_test, nnodes=nnodes, pattern=pattern)
            plt.plot(pred, label=f"nnodes={nnodes}", ls="--")
            error_lists.append(error)
        plt.title("Fitting curve of Batch RBF")
        plt.legend()
        plt.show()
    else:
        for nnodes in nnodes_list:
            pred, error = rbf_delta(x_train, y_train, x_test, y_test, nnodes=nnodes, pattern=pattern)
            plt.plot(pred, label=f"nnodes={nnodes}", ls="--")
            error_lists.append(error)
        plt.title("Fitting curve of Delta on-line RBF")
        plt.legend()
        plt.show()
    return error_lists


# change by variance (sigma)
def rbf_sigma():
    return



if __name__ == '__main__':
    pattern = "square"
    x_train, y_train = data_gen(pattern=pattern, noise=True)
    x_test, y_test = data_gen(start=0.05, pattern=pattern, noise=True)

    # plot the error change by nnodes, delta
    nnodes_list = [i for i in range(2, 11)]
    error_lists = rbf_nnodes(nnodes_list, x_train, y_train, x_test, y_test, pattern=pattern)
    # plot_error(error_lists, "batch RBF")

    error_lists = rbf_nnodes(nnodes_list, x_train, y_train, x_test, y_test, pattern=pattern, rbf="delta")
    # plot_error(error_lists, "delta on-line RBF")


