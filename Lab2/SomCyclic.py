#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
# @author:Clarky Clark Wang
# @license: Apache Licence 
# @file: SomCyclic.py 
# @time: 2021/02/03
# @contact: wangz@kth,se
# @software: PyCharm 
# Import Libs and Let's get started, shall we?
import numpy as np
import os
import matplotlib.pyplot as plt


class SomCity(object):
    def __init__(self, data, w):
        self.x = data
        self.n_points = data.shape[0]
        self.n_weights = w.shape[0]
        self.n_features = data.shape[1]
        self.weight_grid = w

    def find_nearest_weight(self, prop):
        ''' find closets weight to a given input prop returns: (idx, distance) = (int, float)'''
        distances = np.linalg.norm(prop - self.weight_grid, axis=1)
        arg_min = distances.argmin()
        return arg_min, distances[arg_min]

    def get_neighbours(self, idx, radius=1):
        weight_idxes = np.arange(self.n_weights)
        look_around = int(np.floor(radius / 2))
        if (look_around < 1):
            return []
        if (idx == 0):
            return [weight_idxes[9], weight_idxes[1]]
        if (idx == 9):
            return [weight_idxes[0], weight_idxes[8]]

        behind = max(0, idx - look_around)
        infront = min(idx + look_around, len(weight_idxes))

        neighbours = weight_idxes[behind: infront]
        return neighbours[neighbours != idx]

    def train(self, epochs, eta, radius):
        radius_smoothing = 10
        for epoch in range(epochs):
            for i, prop in enumerate(self.x):
                nb, dist = self.find_nearest_weight(prop)
                self.weight_grid[nb] += eta * (prop - self.weight_grid[nb])

                ''' get neighbours around the winning weight '''
                neighbours_idx = self.get_neighbours(idx=nb, radius=radius)
                neighbours = self.weight_grid[neighbours_idx]

                self.weight_grid[neighbours_idx] += eta * (prop - neighbours)

                ''' update learning rate '''
                radius = int(np.ceil(2 * np.exp(-epoch / radius_smoothing)))

    def result(self):
        pos = []
        for i, prop in enumerate(self.x):
            nb, dist = self.find_nearest_weight(prop)
            pos.append(nb)
        return self.x[pos]

def cities():
    cities_path = 'data/cities.dat'

    with open(cities_path) as f:
        data = f.readlines()
        data = [s.rstrip('\n;').split(',') for s in data[4:]]

    return np.array([np.array((float(x), float(y))) for (x, y) in data])



if __name__ == '__main__':
    city= cities()
    print(city)
    w = np.random.uniform(0, 1, (10, 2))
    som_city = SomCity(city, w)
    som_city.train(20, 0.2, 2)
    print(som_city.result())
    tour = som_city.weight_grid
    x = tour[:, 0]
    y = tour[:, 1]

    con_x = [x[0], x[-1]]
    con_y = [y[0], y[-1]]

    plt.scatter(x[0], y[0], label='Start', s=800, facecolors='none', edgecolors='r')
    plt.scatter(x[-1], y[-1], label='End', s=800, facecolors='none', edgecolors='b')
    plt.plot(x, y, 'r', label='Suggested tour')
    plt.plot(con_x, con_y, 'r')  # connect last and first city lol
    plt.scatter(x, y, s=100, c='b', label='Approximated cities')
    plt.scatter(city[:, 0], city[:, 1], marker=('x'), s=150, label='Actual cities')
    plt.legend()
    plt.xlabel('X coordinates', fontsize=18)
    plt.ylabel('Y coordinates', fontsize=18)

    plt.show(block=True)