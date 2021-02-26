#!/usr/bin/env python  
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence 
# @file: DataGen.py 
# @time: 2021/02/14
# @contact: mingj@kth,se
# @software: PyCharm 
# May the Force be with you.
import numpy as np


class Data(object):
    def get_data(self, type="original"):
        if type == "original":
            X = np.array([[-1, -1, 1, -1, 1, -1, -1, 1],
                          [-1, -1, -1, -1, -1, 1, -1, -1],
                          [-1, 1, 1, -1, -1, 1, -1, 1]], dtype=np.int8)
        elif type == "distorted":
            X = np.array([[1, -1, 1, -1, 1, -1, -1, 1],
                          [1, 1, -1, -1, -1, 1, -1, -1],
                          [1, 1, 1, -1, 1, 1, -1, 1]], dtype=np.int8)
        elif type == "more errors":
            X = np.array([[1, 1, -1, -1, -1, 1, -1, 1],
                          [1, 1, 1, -1, -1, -1, 1, -1],
                          [1, -1, -1, 1, -1, 1, -1, -1]], dtype=np.int8)
        elif type == "pic":
            X = self.load_data()
        return X

    def load_data(self):
        pic_data = np.genfromtxt('pict.dat', delimiter=',',
                                 dtype=np.int8).reshape(-1, 1024)
        return pic_data
