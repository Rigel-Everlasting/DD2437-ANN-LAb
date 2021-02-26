from DoubleLayer import *


# generate the data as Gauss function
def f(X, Y):
    e = -(np.square(X) + np.square(Y)) / 10
    Z = np.exp(e) - 0.5
    return Z


def grid_xy(X, Y):
    data = np.vstack((X, Y))
    return data


def generate_data(start, end, step):
    """
    generates X, Y square grid and reshapes it in vectors of size (1, n*m)
    where n = nb lines, m = nb columns
    :param start: grid
    :param end: grid
    :param step:
    :return:
    """
    X, Y = np.mgrid[start:end:step, start:end:step]
    Z = np.exp(-(X ** 2 + Y ** 2) / 10 - 0.5)
    n, m = np.shape(X)
    ndata = n * m
    Xi = X.reshape(1, ndata)
    Yi = Y.reshape(1, ndata)
    targets = Z.reshape(ndata)

    patterns = np.concatenate((Xi, Yi))

    patterns = np.concatenate((patterns, np.ones((1, ndata))))
    return X, Y, Z, patterns, targets


def plot_f(fig, X, Y, Z, title):
    """
    plots approximated function
    :param X:
    :param Y:
    :param Z:
    :return:
    """
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    plt.title(title)


# X = np.random.normal(loc=20, scale=20, size=200)
# Y = np.random.normal(loc=20, scale=20, size=200)
# Z = f(X, Y)
# # fig = plt.figure()
# # plot_f(fig, X, Y, Z, "Gaussian function")
# data = grid_xy(X, Y)
if __name__ == '__main__':
    # X = np.random.randn(1, 10)
    X, Y, Z, patterns, targets = generate_data(-10, 10, 1)
    # fig = plt.figure()
    # plot_f(fig, X, Y, Z, "Original Gaussian function")
    # fig.show()

    # mlp = MLP(epochs=200, Nhidden=20, learning_rate=0.01)
    # Nhidden = mlp.Nhidden
    # init_w1 = np.random.randn(Nhidden, np.shape(patterns)[0])
    # init_w2 = np.random.randn(1, Nhidden + 1)
    #
    # w1, w2, predictions, errors, misclassification_ratios = \
    #     mlp.back_propagation(patterns, targets, init_w1, init_w2, binary=False)
    # # plot the error
    # plot_error(errors, "2-Layers-BPNN, for the Gaussian Function")
    # print(np.mean(errors))
    # n, m = X.shape
    # pred_Z = predictions.reshape((n, m))
    # fig = plt.figure()
    # plot_f(fig, X, Y, pred_Z, "Modelling Gaussian function")
    # fig.show()

    # split the data
    ratio_list = [0.2, 0.4, 0.6, 0.8]
    training_errors, all_errors, pred_Z = errors_of_all(patterns, targets, ratio_list)

    # errors changed by split_ratio
    for i in range(len(ratio_list)):
        plt.subplot(2, 2, i + 1)
        plt.plot(training_errors[i], label="training")
        plt.plot(all_errors[i], label="all data")
        plt.xlabel = ("epochs")
        plt.ylabel("errors")
        plt.title(f'split ratio={ratio_list[i]}')
        plt.legend()
    plt.suptitle("error curve changed by split ratio")
    plt.show()
    for i in range(len(ratio_list)):
        fig = plt.figure()
        # print(X.shape, Y.shape, pred_Z[i].shape)
        plot_f(fig, X, Y, pred_Z[i].reshape(X.shape), f"Modelling curve of split ratio{ratio_list[i]}")
        fig.show()