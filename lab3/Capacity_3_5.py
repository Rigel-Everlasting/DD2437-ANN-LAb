# ANN-course, Emil
# "A dark-themed editor is a pathway to many abilities some consider unnatural"

import numpy as np
from HopfieldNN import *
from Distortion_Resistance_3_4 import *


def get_retrievable_patterns(X, model, bits_to_flip=None, mu=None, diagonal=None):
    """Tests how well a given model stores patterns.
    Computes the NUMBER OF retrievable (or stable) patterns as more patterns are added to the model one by one."""
    N = X.shape[0]; d = X.shape[1]

    print('Running...')

    retrievable_patterns = []
    for i in range(1, N + 1):
        # print('Processing %d:th pattern' %(i))
        X_current = X[0:i, :]
        model.train(X_current, mu=mu, diagonal=diagonal)  # this is a bit cheap but is equivalent to training HopfieldNN online
        if bits_to_flip:
            X_flipped = X_current.astype(np.float64).copy()
            for j in range(i):
                indices_to_flip = np.random.choice(d, bits_to_flip, replace=False)
                X_flipped[j, indices_to_flip] = X_current[j, indices_to_flip] * (-1)
            X_test = X_flipped
        else:
            X_test = X_current.astype(np.float64).copy()

        check = check_res(model, X_current, X_test)
        count = np.count_nonzero(check)
        retrievable_patterns.append(count)

    return retrievable_patterns


if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------#
    # 3.5.1 - EVERYTHING IS JUST COPIED FROM JAMIE's code for 3_4
    data = Data()
    pics = data.get_data("pic")
    mode = "synchronous"
    iter = 100

    X = pics[1:4, :]  # just change the number
    hn = HopfieldNetwork()
    hn.train(X)
    hn.check_memory(X)

    pics = pics[1:4, :]
    percentage_list = np.linspace(start=0, stop=1, num=101)
    noised_recall(hn, pics, percentage_list, num=12, plot=True)

    # # ---------------------------------------------------------------------------------------#
    # # 3.5.2 - EVERYTHING IS BASICALLY JUST COPIED FROM JAMIE's code for 3_4
    # mode = "synchronous"; iter = 100
    #
    # d = 1024; N = 50  # change N
    # X = np.random.randint(2, size=d * N).reshape(N, d)
    # X[X == 0] = -1
    #
    # hn = HopfieldNetwork()
    # hn.train(X)
    # hn.check_memory(X)
    #
    # pics = X
    # percentage_list = np.linspace(start=0, stop=1, num=21)
    # noised_recall(hn, pics, percentage_list, num=100, plot=True)
    #
    # # ---------------------------------------------------------------------------------------#
    # # 3.5.3
    # # Theory question, see file.
    #
    # # ---------------------------------------------------------------------------------------#
    # # 3.5.4
    # d = 100; N = 1000
    # np.random.seed(22)
    # X = np.random.randint(2, size=d*N).reshape(N,d)
    # X[X==0] = -1
    #
    # hn = HopfieldNetwork()
    # no_stable_patterns = get_retrievable_patterns(X, hn)
    # fig, ax = plt.subplots()
    # ax.plot(range(1, N + 1), no_stable_patterns, color='Navy')
    # ax.set_xlabel('No. of patterns stored')
    # ax.set_ylabel('No. of stable patterns')
    # ax.set_title('Without noise but with self-connections in W')
    # fig.show()
    #
    # # ---------------------------------------------------------------------------------------#
    # # 3.5.5
    # d = 100; N = 300; bits_to_flip = 10
    # np.random.seed(22)
    # X = np.random.randint(2, size=d*N).reshape(N ,d)
    # X[X==0] = -1
    #
    # hn = HopfieldNetwork()
    # correct_converged_patterns = get_retrievable_patterns(X, hn, bits_to_flip)
    # fig, ax = plt.subplots()
    # ax.plot(range(1, N + 1), correct_converged_patterns, color='Navy')
    # ax.set_xlabel('No. of patterns stored')
    # ax.set_ylabel('No. of converging patterns')
    # ax.set_title('With noise but with self-connections in W')
    # fig.show()
    #
    # # ---------------------------------------------------------------------------------------#
    # # 3.5.6
    # d = 100; N = 300; bits_to_flip = 10
    # np.random.seed(22)
    # X = np.random.randint(2, size=d*N).reshape(N,d)
    # X[X==0] = -1
    #
    # hn = HopfieldNetwork()
    # no_stable_patterns = get_retrievable_patterns(X, hn, bits_to_flip, diagonal="diag0")    # remove bits_to_flip to see what they're saying in the first paragraph.
    # fig, ax = plt.subplots()
    # ax.plot(range(1, N + 1), no_stable_patterns, color='Navy')
    # ax.set_xlabel('No. of patterns stored')
    # ax.set_ylabel('No. of stable patterns')
    # ax.set_title('With Noise and without self-connections in W')
    # fig.show()
    #
    # # ---------------------------------------------------------------------------------------#
    # # 3.5.7
    # d = 100; N = 300; bits_to_flip = 10; data_bias = 0.5
    # np.random.seed(22)
    # X = np.random.normal(loc=0, scale=1, size=(N,d)) + data_bias
    # X = np.sign(X)
    #
    # hn = HopfieldNetwork()
    # no_stable_patterns = get_retrievable_patterns(X, hn, bits_to_flip, diagonal="diag0")    # remove bits_to_flip to see what they're saying in the first paragraph.
    # fig, ax = plt.subplots()
    # ax.plot(range(1, N + 1), no_stable_patterns, color='Navy')
    # ax.set_xlabel('No. of patterns stored')
    # ax.set_ylabel('No. of stable patterns')
    # ax.set_title('With biased data, Noise and without self-connections in W')
    # fig.show()
    #
    # # input('Showing plots, press any key to exit script')
