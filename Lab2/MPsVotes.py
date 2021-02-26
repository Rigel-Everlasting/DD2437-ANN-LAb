#!/usr/bin/env python  
# -*- coding:utf-8 -*-
# @author: JamieJ
# @license: Apache Licence 
# @file: MPsVotes.py 
# @time: 2021/02/06
# @contact: mingj@kth,se
# @software: PyCharm 
# May the Force be with you.
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    votes = np.loadtxt("data_lab2/votes.dat", comments="%", delimiter=',').reshape((349, 31))
    gender = np.loadtxt("data_lab2/mpsex.dat", comments="%")
    district = np.loadtxt("data_lab2/mpdistrict.dat", comments="%")
    party = np.loadtxt("data_lab2/mpparty.dat", comments="%")
    attrs = {'gender': gender, "district": district, "party": party}
    return votes, attrs


class MPs(object):
    def __init__(self, votes, w):
        self.x = votes
        self.weight_grid = w
        self.n_points = votes.shape[0]
        self.n_features = votes.shape[1]
        self.n_weights = w.shape[0]

    def find_nearest_weight(self, prop):
        # ord = 1, Manh distance
        distances = np.linalg.norm(prop - self.weight_grid, axis=1, ord=1)
        arg_min = distances.argmin()
        return arg_min, distances[arg_min]

    def get_neighbours(self, idx, radius=1):
        weight_idxes = np.arange(self.n_weights)
        look_around = int(np.floor(radius / 2))
        behind = max(0, idx - look_around)
        infront = min(idx + look_around, len(weight_idxes))
        neighbours = weight_idxes[behind: infront]
        return neighbours[neighbours != idx]

    def train(self, epochs, eta, radius):
        radius_smoothing = 2
        for epoch in range(epochs):
            for i, prop in enumerate(self.x):
                nb, dist = self.find_nearest_weight(prop)
                self.weight_grid[nb] += eta * (prop - self.weight_grid[nb])

                neighbours_idx = self.get_neighbours(idx=nb, radius=radius)
                neighbours = self.weight_grid[neighbours_idx]

                self.weight_grid[neighbours_idx] += eta * (prop - neighbours)

                radius = int(np.ceil(epochs * np.exp(-epoch / radius_smoothing)))

    def result(self):
        position = []
        for i, prop in enumerate(self.x):
            nb, dist = self.find_nearest_weight(prop)
            position.append(nb)
        return np.array(position)


def add_noise(point, noise=0.3):
    point = point.astype(float)
    point += np.random.uniform(-noise, noise, size=point.shape)
    return point


def plot_with_attr(pos, attrs, attr_name, gridsize=(10, 10)):
    attr_list = np.unique(attrs)
    for attr in attr_list:
        x, y = np.unravel_index(pos[attrs == attr], gridsize)

        # add some noise to x, y to avoid overlapping
        plt.scatter(add_noise(x), add_noise(y), label=f"{attr_name}{int(attr)}")
    plt.title(f"{attr_name}")
    plt.grid(True)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


if __name__ == '__main__':
    votes, attrs = load_data()
    w = np.random.uniform(0, 1, (100, votes.shape[1]))
    model = MPs(votes, w)
    params = {
        'epochs': 100,
        'eta': 0.2,
        'radius': 50
    }
    model.train(params['epochs'], params['eta'], params['radius'])
    position = model.result()
    for attr_name, attrs in attrs.items():
        plot_with_attr(position, attrs=attrs, attr_name=attr_name)

