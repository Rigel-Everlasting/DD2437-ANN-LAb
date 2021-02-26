#!/usr/bin/env python  
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence 
# @file: Sequential_Update_3_2.py
# @time: 2021/02/14
# @contact: mingj@kth,se
# @software: PyCharm 
# May the Force be with you.
from HopfieldNN import *
import math


def pics_display(pics):
    l = len(pics)
    row = math.ceil(math.sqrt(l))
    col = int(math.sqrt(l))
    for i in range(l):
        pic = pics[i]
        x_y = pic.reshape((32, 32))
        plt.subplot(row, col, i + 1)
        plt.imshow(x_y, cmap="gray")
    plt.show()


def compare(model, pic):
    plt.subplot(1, 2, 1)
    plt.imshow(pic.reshape((32, 32)), cmap="gray")
    # recalled = model.recall(pic)
    recalled = model.recall(pic, mode="asynchronous")
    plt.subplot(1, 2, 2)
    plt.imshow(recalled.reshape((32, 32)), cmap="gray")
    plt.show()
    return recalled


def sequential(model, pic, iter_list):
    start = 0
    for i in range(len(iter_list)):
        plt.subplot(1, len(iter_list), i + 1)
        updated_pic = pic
        # updated_pic = model.recall(updated_pic, mode="asynchronous", iter=iter_list[i] - start)
        for _ in range(start, iter_list[i]):
            updated_pic = model.update(updated_pic, mode="asynchronous")
        pic = updated_pic
        plt.imshow(pic.reshape((32, 32)), cmap="gray")
        plt.title(f"iteration={iter_list[i]}")
        start = iter_list[i]
    plt.suptitle("Sequential Convergence Process")
    plt.show()


if __name__ == '__main__':
    data = Data()
    pics = data.get_data("pic")
    mode = "synchronous"
    iter = 100

    # ---------------------------------------------------------------------------------------#
    # # display the pictures data
    # pics_display(pics)

    X = pics[:3, :]
    hn = HopfieldNetwork()
    hn.train(X)
    hn.check_memory(X)

    # ---------------------------------------------------------------------------------------#
    # 3.2-2
    p10 = pics[9]
    p11 = pics[10]
    # compare(hn, p10)
    # compare(hn, p11)

    # ---------------------------------------------------------------------------------------#
    # 3.2-3
    iter_list = range(256, 1025, 256)
    sequential(hn, p10, iter_list)
