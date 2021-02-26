#!/usr/bin/env python  
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence 
# @file: Distortion_Resistance_3_4.py
# @time: 2021/02/16
# @contact: mingj@kth,se
# @software: PyCharm 
# May the Force be with you.

from HopfieldNN import *
import random


def noised_pic(pics, percentage_list):
    """
     add noise to the ALL pics dataset
    :param pics: a list, include original pics dataset
    :param percentage_list: a list, noised bits percentage
    :return noised data: ALL noised data, but for each percentage_list only one noised data is generated.
    """
    noised_data = []
    for pic in pics:
        noised_pic = []
        for i, pct in enumerate(percentage_list):
            noised = np.copy(pic)
            noised_bits = int(pic.shape[0] * pct)
            noised_index = random.sample(range(0, pic.shape[0]), noised_bits)
            noised[noised_index] = -noised[noised_index]
            noised_pic.append(noised)
        noised_data.append(noised_pic)
    return noised_data


# add the noise with the given percentage on a picture
# for this percentage, generate as much noised data as the give params num
# return a list, containing the noised data for this picture with this noise percentage
def add_noise(pic, noise_percentage, num=1):
    noised_in_this_pctg = []
    for _ in range(num):
        noised = np.copy(pic)
        noised_bits = int(pic.shape[0] * noise_percentage)
        noised_index = random.sample(range(0, pic.shape[0]), noised_bits)
        noised[noised_index] = -noised[noised_index]
        noised_in_this_pctg.append(noised)
    return noised_in_this_pctg


# return the noised picture with the noise percentage in the list
# with each percentage, the noised data number is decided by the param num
def noised_by_pctg(pic, percentage_list, num=1):
    noised_pic = []
    for pctg in percentage_list:
        noised_pic.append(add_noise(pic, pctg, num=num))
    return noised_pic


def noised_recall(model, pics, percentage_list, num=100, plot=True, iter=10):
    good_recalls = []
    for i, pic in enumerate(pics):
        noised_pic = noised_by_pctg(pic, percentage_list, num=num)

        good_recalls_by_pctg = []
        for noised_in_this_pctg in noised_pic:
            # check all of the noised pic with one noised percentage
            # the number equals the param num
            # record the recalled that converge to the attractors
            good_recalls_in_this_pctg = 0
            for noised in noised_in_this_pctg:
                recalled = model.recall(noised, iter=iter)

                # check if this is a good recall
                if np.array_equal(recalled, pic):
                    good_recalls_in_this_pctg += 1

            good_recalls_by_pctg.append(good_recalls_in_this_pctg)
        good_recalls.append(good_recalls_by_pctg)
        if plot:
            plt.plot(percentage_list, good_recalls_by_pctg, label=f"pic{i + 1}")
    if plot:
        plt.xlabel = ("noise percentage")
        plt.ylabel(f"good recall counts in {num} noised data")
        plt.title("Performance Changed by Noise Percentage")
        plt.legend()
        plt.show()
    return good_recalls


def plot_compare(model, pic, noised_pics, percentages, iter=10):
    col = len(noised_pics)+1
    plt.subplot(1, col, 1)
    plt.imshow(pic.reshape((32, 32)), cmap="gray")
    for i, noised_pic in enumerate(noised_pics):
        recalled = model.recall(noised_pic, iter=iter)
        x_y = recalled.reshape((32, 32))
        plt.subplot(1, col, i + 2)
        plt.imshow(x_y, cmap="gray")
        plt.title(f"{percentages[i]} noise")
    plt.legend()
    plt.suptitle("p1 restoring performance with different noise percentage")
    plt.show()


if __name__ == '__main__':
    data = Data()
    pics = data.get_data("pic")
    mode = "synchronous"
    iter = 100

    # ---------------------------------------------------------------------------------------#
    # train HNN with p1, p2 and p3
    X = pics[:3, :]
    hn = HopfieldNetwork()
    hn.train(X)
    hn.check_memory(X)

    # add noise to p1 p2 and p3
    pics = X
    percentage_list = np.linspace(start=0, stop=1, num=101)
    # # plot the accuracy
    # noised_data = noised_pic(pics, percentage_list)
    # plot_noised(pics, noised_data, percentage_list)

    # ---------------------------------------------------------------------------------------#
    # plot the good recalls counts
    noised_recall(hn, pics, percentage_list, num=100, plot=True, iter=20)

    # ---------------------------------------------------------------------------------------#
    # plot the restoring results
    p1 = pics[0]
    percentages = [0.2, 0.5, 0.8]
    noised_p1 = noised_pic([p1], percentages)[0]
    plot_compare(hn, p1, noised_p1, percentages, iter=10)

# ---------------------------------------------------------------------------------------#
# former version

# def accuracy(y, pred):
#     same = np.where(y == pred)[0].shape[0]
#     return same / y.shape[0]
#
#
# def check_restored(model, pic, noised_pics):
#     # train the model with the pure pic data
#     # model = HopfieldNetwork()
#     # model.train([pic])
#     recalled, acc = [], []
#     for noised in noised_pics:
#         recalled_noised = model.recall(noised, iter=10)
#         recalled.append(recalled_noised)
#         acc.append(accuracy(pic, recalled_noised))
#     return recalled, acc
#
#
# def plot_noised(pics, noised_data, percentage_list):
#     accs = []
#     model = HopfieldNetwork()
#     model.train(pics)
#     for i, pic in enumerate(pics):
#         noised_pics = noised_data[i]
#         recalled, acc = check_restored(model, pic, noised_pics)
#         accs.append(acc)
#         plt.plot(percentage_list, acc, label=f"pic{i + 1}")
#     plt.title("3-4")
#     plt.legend()
#     plt.show()
#     return accs
