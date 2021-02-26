#!/usr/bin/env python  
# -*- coding:utf-8 _*-
# @author:Clarky Clark Wang
# @license: Apache Licence 
# @file: SomAnimal.py 
# @time: 2021/02/03
# @contact: wangz@kth,se
# @software: PyCharm 
# Import Libs and Let's get started, shall we?
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class SomAnimal(object):
    def __init__(self, animal_data, animal_name, w):
        # self.animal_data = np.loadtxt('data/animals.dat', dtype='i', delimiter=',').reshape((32, 84))
        # self.animal_name = np.loadtxt('data/animalnames.txt', delimiter='\t\n', dtype='<U11').reshape((32, 1))
        # self.w = np.random.uniform(0, 1, (100, 84))
        self.x = animal_data
        self.n_points = animal_data.shape[0]
        self.n_weights = w.shape[0]
        self.n_features = animal_data.shape[1]
        self.weight_grid = w
        self.animal_name = animal_name

    def find_nearest_weight(self, prop):
        distances = np.linalg.norm(prop - self.weight_grid, axis=1)
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
        pos = []
        pos_vector = np.zeros(self.n_points)
        for i, prop in enumerate(self.x):
            nb, dist = self.find_nearest_weight(prop)
            pos.append(nb)
            print(dist)
        return self.animal_name[np.argsort(pos)].reshape(32)


if __name__ == '__main__':
    animal_data = np.loadtxt('data_lab2/animals.dat', dtype='i', delimiter=',').reshape((32, 84))
    animal_name = np.loadtxt('data_lab2/animalnames.txt', delimiter='\t\n', dtype='<U11').reshape((32, 1))
    w = np.random.uniform(0, 1, (100, 84))
    model = SomAnimal(animal_data, animal_name, w)
    settings = {
        'epochs': 20,
        'eta': 0.2,
        'radius': 50
    }
    model.train(settings['epochs'], settings['eta'], settings['radius'])
    print(model.result())
