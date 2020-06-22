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
    array = np.zeros((n * cluster_seeds.shape[0], dim))
    index = 0
    for seed in cluster_seeds:
        for i in range(n):
            array[index] += seed + np.random.choice([-1, 1]) * max_dif * np.random.random(dim)
            index += 1

    return array


# noinspection PyArgumentList
class KMeansClassifier:
    def __init__(self, k: int, train_data: np.ndarray, norm: int = 2, method="Nelder–Mead", cb0=None):
        self.k = k
        self.train_data = train_data
        self.norm = norm
        self.method = method

        self.init_code_book = cb0 if cb0 is not None else self.__generate_init_code_book(k, train_data.shape[1],
                                                                                         train_data.min(),
                                                                                         tra_data.max())

        self.assignment_table = self.__generate_assignment_table(self.init_code_book, self.train_data)

    @staticmethod
    def __generate_init_code_book(k, dim, min_val, max_val):
        array = np.zeros((k, dim))
        for k_i in range(k):
            array[k_i] += np.random.choice([min_val, max_val]) * np.random.random(dim)
        return array

    def __generate_assignment_table(self, cluster_matrix: np.ndarray, data_matrix: np.ndarray):
        m = np.zeros((data_matrix.shape[0], cluster_matrix.shape[0]))
        for row_i, row in enumerate(m):
            ld_index = self.__get_lowest_distance_index(data_matrix[row_i], cluster_matrix)
            row[ld_index] = 1
        return m

    def __get_lowest_distance_index(self, data_vector: np.ndarray, cluster_matrix: np.ndarray):
        lowest_distance = None
        lowest_distance_index = None
        for cluster_index, cluster_v in enumerate(cluster_matrix):
            tmp = np.linalg.norm(data_vector - cluster_v, ord=self.norm)
            if lowest_distance is None or tmp < lowest_distance:
                lowest_distance = tmp
                lowest_distance_index = cluster_index
        return lowest_distance_index



if __name__ == '__main__':
    # set loglevel
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    # init plot
    plt.style.use("dark_background")
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)

    # setup test data
    cl_se = np.array([[1, 1],
                      [1, 6]])
    tra_data = generate_test_data(cl_se, 6, 1)

    # generate color dict
    labels = np.unique(list(range(cl_se.shape[0])))
    colors = np.random.choice(list(mpl_colors.XKCD_COLORS.keys()), size=len(labels))
    color_dict = dict(zip(labels, colors))
    legend_patches = [patches.Patch(color=color_dict[key], label=key) for key in color_dict.keys()]

    # init classifier
    classifier = KMeansClassifier(cl_se.shape[0], tra_data)

    for i, label in enumerate(labels):
        legend_patches.append(ax0.plot(cl_se[i, 0],
                                       cl_se[i, 1],
                                       linestyle=" ",
                                       marker='o',
                                       color=colors[label],
                                       label=f"cluster seed {label}")[0])

        legend_patches.append(ax0.plot(classifier.init_code_book[i, 0],
                                       classifier.init_code_book[i, 1],
                                       linestyle=" ",
                                       marker='^',
                                       color=colors[label],
                                       label=f"code book vector {label}")[0])

        legend_patches.append(ax0.plot(classifier.train_data[classifier.assignment_table[:, i] == 1, 0],
                                       classifier.train_data[classifier.assignment_table[:, i] == 1, 1],
                                       linestyle=" ",
                                       marker='.',
                                       color=colors[label],
                                       label=f"train data assigned to {label}")[0])



    # plot newly classified data
    plt.legend(handles=legend_patches)
    plt.show()
