#!/usr/bin/env python  
#-*- coding:utf-8 -*-  
# @author:Clarky Clark Wang
# @license: Apache Licence 
# @file: Sparse_3_6.py
# @time: 2021/02/17
# @contact: wangz@kth,se
# @software: PyCharm 
# Import Libs and Let's get started, shall we?
from HopfieldNN import *


def num_pattern_training(theta_list, iter, pattern_size=300, pattern_activity=0.1):
    for theta in theta_list:
        model = HopfieldNetwork(bias=theta)
        successes_by_num_patterns = []
        for n in range(1, 21):
            model_success = 0
            for _ in range(iter):
                n_ones = int(pattern_activity * pattern_size * n)
                patterns = np.zeros(n * pattern_size, dtype=np.int8)
                indexes_ones = np.random.choice(len(patterns), size=n_ones, replace=False)
                patterns[indexes_ones] = 1
                patterns = np.hsplit(patterns, n)
                model.train(patterns, mu=pattern_activity)
                recalled = model.recall(patterns)
                if np.array_equal(recalled, patterns):
                    model_success += 1
            successes_by_num_patterns.append(model_success)
        plt.plot(range(1, 21), successes_by_num_patterns, label=r"$\Theta = {:.3f}$".format(theta))
    plt.xlabel("Number of patterns for training")
    plt.ylabel("Percent of success: all training patterns well memorised")
    plt.legend()
    plt.xticks(range(1, 21))
    plt.show()





if __name__ == '__main__':
    THETA = np.arange(0, 0.08, 0.02)
    iter = 40
    num_pattern_training(theta_list=THETA, iter=iter, pattern_activity=0.1)
    num_pattern_training(theta_list=THETA, iter=iter, pattern_activity=0.05)