#!/usr/bin/env python  
#-*- coding:utf-8 -*-
# @author:Clarky Clark Wang
# @license: Apache Licence
# @file: 3_3_Energy.py
# @time: 2021/02/16
# @contact: wangz@kth,se
# @software: PyCharm
# Import Libs and Let's get started, shall we?
from HopfieldNN import *

def sequential_update(train_pattern, test_pattern, mode, iter):
    model = HopfieldNetwork()
    model.train(train_pattern)
    pattern_update = test_pattern
    energy_list = []
    for _ in range(iter):
        pattern_update = model.update(pattern_update, mode=mode)
        energy_list.append(model.energy(pattern_update))
    x = range(1, len(energy_list)+1)
    plt.plot(x, energy_list)
    plt.show()

def random_weight(test_pattern, mode, iter):
    model = HopfieldNetwork()
    model.weight = np.random.randn(1024, 1024)
    pattern_update = test_pattern
    energy_list = []
    for _ in range(iter):
        pattern_update = model.update(pattern_update, mode = mode)
        energy_list.append(model.energy(pattern_update))
    x = range(1, len(energy_list) + 1)
    plt.plot(x, energy_list)
    plt.show()


def sys_weight(train_pattern, test_pattern, mode, iter):
    model = HopfieldNetwork()
    model.train(train_pattern, symmetric=True)
    energy_list = []
    pattern_update = test_pattern
    for _ in range(iter):
        pattern_update = model.update(pattern_update, mode=mode)
        energy_list.append(model.energy(pattern_update))
    x = range(1, len(energy_list) + 1)
    plt.plot(x, energy_list)
    plt.show()





if __name__ == '__main__':
    data = Data()
    X, distorted_X = data.get_data(), data.get_data("distorted")
    hn = HopfieldNetwork()
    hn.train(X)
    attractors = hn.attractors_count()
    ### Question 3.1.1-3.1.2
    print(f"{len(attractors)} attractors are found.")
    for attractor in attractors:
        print("{} => {}".format(attractor, hn.energy(attractor)))
    print('For Distorted Patterns:')
    for pattern in distorted_X:
        print("{} => {}".format(pattern, hn.energy(pattern)))
    ### Question 3.1.3
    pics = data.load_data()
    iter = 1024
    mode = "asynchronous"
    train_pattern, test_pattern = pics[0:3, :], pics[9, :]
    sequential_update(train_pattern, test_pattern, mode, iter)
    ### Question 3.1.4
    random_weight(test_pattern, mode, iter)
    ### Question 3.1.5
    sys_weight(train_pattern, test_pattern, mode, iter)






