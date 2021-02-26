#!/usr/bin/env python  
# -*- coding:utf-8 _*-
# @author: JamieJ
# @license: Apache Licence 
# @file: DoubleLayer.py 
# @time: 2021/01/23
# @contact: mingj@kth,se
# @software: PyCharm 
# May the Force be with you.
from DataGen import *
from SingleLayer import *
import matplotlib.colors as color

'''
X: inputs 
T: true labels, [1, -1]
W: initial weight matrix of first layer
V: initial weight matrix of second layer
returns: H_out: outputs of the first layer
         O_out: outputs of the second layer
'''


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


def plot_error(errors, title):
    plt.plot(errors)
    plt.title(f'Learning error curve for the {title}')
    plt.xlabel = ("Epochs")
    plt.ylabel("Error")
    plt.show()


def plot_mis(mis, title):
    plt.plot(mis)
    plt.title(f'Misclassifications curve for the {title}')
    plt.xlabel = ("Epochs")
    plt.ylabel("Misclassifications_ratio")
    plt.show()


def plot_axis(X, h=0.1):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_data = np.transpose(np.c_[xx.ravel(), yy.ravel()])
    ones = np.transpose(np.ones((np.shape(xx)[1] * np.shape(xx)[0], 1)))
    grid_data = np.vstack((grid_data, ones))
    return xx, yy, grid_data


def plot_boundary(xx, yy, predictions, title):
    Z = predictions
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.twilight, alpha=0.8)
    plt.scatter(X[0, :], X[1, :], c=T, cmap=color.ListedColormap(['red', 'blue']))
    plt.title(f'Boundary curve for the {title}')
    plt.show()


def plot_comparison(col, row, params_list, y_list, xlabel, ylabel, title):
    for i in range(len(y_list)):
        plt.subplot(col, row, i + 1)
        plt.plot(y_list[i])
        plt.xlabel = (f"{xlabel}")
        plt.ylabel(f"{ylabel}")
        plt.title(f'{xlabel}={params_list[i]}')
    plt.suptitle(f'{ylabel} changed by {xlabel} of the {title}')
    plt.show()


def split(X, T, split_ratio=0.2):
    nums = T.shape[0]
    training_index = nums - int(nums * split_ratio)
    x_training = X[:, :training_index]
    y_training = T[:training_index]
    x_val = X[:, training_index:]
    y_val = T[training_index:]
    return x_training, y_training, x_val, y_val


def validation(X, T, ratio_list):
    training_errors = []
    traning_mises = []
    val_errors = []
    val_mises = []
    for ratio in ratio_list:
        x_training, y_training, x_val, y_val = split(X, T, split_ratio=ratio)
        mlp = MLP(epochs=300, Nhidden=5)
        Nhidden = mlp.Nhidden
        train_w1 = np.random.randn(Nhidden, np.shape(x_training)[0])
        train_w2 = np.random.randn(1, Nhidden + 1)

        w1, w2, train_predictions, training_error, train_mis_ratio = \
            mlp.back_propagation(x_training, y_training, train_w1, train_w2)

        training_errors.append(training_error)
        traning_mises.append(train_mis_ratio)

        w_list = mlp.w_list
        v_list = mlp.v_list
        val_error = []
        val_mis = []

        for i in range(len(w_list)):
            _, y = mlp.forward_pass(x_val, w_list[i], v_list[i])
            val_error.append(error_compute(y_val, y))
            val_mis.append(misclassification_ratio(y_val, y))
            # # plot the boundary
            # xx, yy, grid_data = plot_axis(X)
            # grid_predictions = mlp.predict(grid_data, w_list[i], v_list[i])
            # plot_boundary(xx, yy, grid_predictions, f"{i}")
        val_errors.append(val_error)
        val_mises.append(val_mis)
        # plot the boundary
        xx, yy, grid_data = plot_axis(X)
        grid_predictions = mlp.predict(grid_data, w_list[-1], v_list[-1])
        plot_boundary(xx, yy, grid_predictions, f"boundary for split ratio {ratio}")

    return training_errors, traning_mises, val_errors, val_mises


def errors_of_all(X, T, ratio_list):
    training_errors = []
    all_errors = []
    predictions = []
    for ratio in ratio_list:
        x_training, y_training, x_val, y_val = split(X, T, split_ratio=ratio)
        mlp = MLP(epochs=1000, Nhidden=10)
        Nhidden = mlp.Nhidden
        train_w1 = np.random.randn(Nhidden, np.shape(x_training)[0])
        train_w2 = np.random.randn(1, Nhidden + 1)

        w1, w2, train_predictions, training_error, train_mis_ratio = \
            mlp.back_propagation(x_training, y_training, train_w1, train_w2)

        training_errors.append(training_error)

        w_list = mlp.w_list
        v_list = mlp.v_list
        all_error = []
        for i in range(len(w_list)):
            prediction = mlp.predict(X, w_list[i], v_list[i], binary=False)
            all_error.append(error_compute(T, prediction))
        all_errors.append(all_error)
        predictions.append(prediction)

    return training_errors, all_errors, predictions


def errors_epochs(epochs_list, X, T):
    errors_list = []
    misclassification_ratios = []
    for epochs in epochs_list:
        mlp = MLP(epochs=epochs)
        _, _, _, errors, misclassification = mlp.back_propagation(X, T, init_w1, init_w2)
        errors_list.append(errors)
        misclassification_ratios.append(misclassification)
    plot_comparison(2, 2, epochs_list, errors_list, "epochs", "errors", "2-Layers-BPNN")
    plot_comparison(2, 2, epochs_list, misclassification_ratios,
                    "epochs", "misclassification", "2-Layers-BPNN")


def errors_nhidden(Nhidden_list, X, T):
    errors_list = []
    misclassification_ratios = []
    for Nhidden in Nhidden_list:
        mlp = MLP(Nhidden=Nhidden, epochs=300)
        init_w1 = np.random.randn(Nhidden, np.shape(X)[0])
        init_w2 = np.random.randn(1, Nhidden + 1)
        _, _, _, errors, misclassification = mlp.back_propagation(X, T, init_w1, init_w2)
        errors_list.append(errors)
        misclassification_ratios.append(misclassification)
    plot_comparison(2, 2, Nhidden_list, errors_list, "Nhiddens", "errors", "2-Layers-BPNN")
    plot_comparison(2, 2, Nhidden_list, misclassification_ratios,
                    "Nhiddens", "misclassification", "2-Layers-BPNN")


if __name__ == '__main__':
    data = DataGenerator()
    classA, classB = data.non_lin_data()
    X, T, Weight = process_input(classA, classB)
    # learning_rate = 0.001
    # alpha = 0.9
    epochs = 300
    mlp = MLP(epochs=epochs, Nhidden=10)
    Nhidden = mlp.Nhidden
    init_w1 = np.random.randn(Nhidden, np.shape(X)[0])
    init_w2 = np.random.randn(1, Nhidden + 1)
    print(X.shape, T.shape, init_w1.shape, init_w2.shape)

    w1, w2, predictions, errors, misclassification_ratios = mlp.back_propagation(X, T, init_w1, init_w2)
    print(errors)

    # # plot the error
    # plot_error(errors, "2-Layers-BPNN")
    # # plot the misclassification ratio
    # plot_mis(misclassification_ratios, "2-Layers-BPNN")

    # # plot the boundary
    # xx, yy, grid_data = plot_axis(X)
    # grid_predictions = mlp.predict(grid_data, w1, w2)
    # plot_boundary(xx, yy, grid_predictions, "2-Layers-BPNN")
    

    # # errors changed by epochs
    # epochs_list = [50, 100, 200, 500]
    # errors_epochs(epochs_list, X, T)


    # # errors changed by Nhidden
    # Nhidden_list = [4, 5, 10, 12]
    # errors_nhidden(Nhidden_list, X, T)


    # # split validation set
    # ratio_list = [0.2, 0.3, 0.5, 0.6]
    # training_errors, traning_mises, val_errors, val_mises = validation(X, T, ratio_list)
    #
    #
    # # errors changed by split_ratio
    # for i in range(len(ratio_list)):
    #     plt.subplot(2, 2, i + 1)
    #     plt.plot(training_errors[i], label="training")
    #     plt.plot(val_errors[i], label="val")
    #     plt.xlabel = ("epochs")
    #     plt.ylabel("errors")
    #     plt.title(f'split ratio={ratio_list[i]}')
    #     plt.legend()
    # plt.suptitle("error curve changed by split ratio")
    # plt.show()
    #
    # # misclassification changed by split_ratio
    # for i in range(len(ratio_list)):
    #     plt.subplot(2, 2, i + 1)
    #     plt.plot(traning_mises[i], label=" training")
    #     plt.plot(val_mises[i], label="val")
    #     plt.xlabel = ("epochs")
    #     plt.ylabel("misclassification")
    #     plt.title(f'split ratio={ratio_list[i]}')
    #     plt.legend()
    # plt.suptitle("misclassification curve changed by split ratio")
    # plt.show()
