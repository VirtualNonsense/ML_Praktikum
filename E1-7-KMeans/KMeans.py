import numpy as np
import pandas as pd
import typing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import logging
import os

from matplotlib import colors as mpl_colors


def generate_test_data(cluster_seeds, n, max_dif):
    dim = cluster_seeds.shape[1]
    array = np.zeros((n*cluster_seeds.shape[0], dim))
    index = 0
    for seed in cluster_seeds:
        for i in range(n):
            array[index] += seed + np.random.choice([-1, 1]) * max_dif * np.random.random(dim)
            index += 1

    return array


if __name__ == '__main__':
    cl_se = np.array([[1, 1],
                      [1, 6]])

    tra_data = generate_test_data(cl_se, 6, 2)



