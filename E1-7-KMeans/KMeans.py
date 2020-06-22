import numpy as np
import pandas as pd
import typing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import logging
import os

from matplotlib import colors as mpl_colors
from scipy.optimize import minimize


def generate_test_data(cluster_seeds: np.ndarray, n, max_dif):
    dim = cluster_seeds.shape[1] if len(cluster_seeds.shape) >= 2 else cluster_seeds.shape[0]
    array = np.zeros((n * cluster_seeds.shape[0], dim))
    index = 0
    for seed in cluster_seeds:
        for i in range(n):
            array[index] += seed + np.random.choice([-1, 1]) * max_dif * np.random.random(dim)
            index += 1

    return array


# noinspection PyArgumentList
class KMeansClassifier:
    def __init__(self, k: int,
                 train_data: np.ndarray,
                 norm: int = 2,
                 method='Nelder-Mead',
                 cb0=None,
                 max_iterations=11,
                 epsilon=.3,
                 allow_code_book_pruning=False):
        """
        This classifier tries to split a given data set into k classes, by assigning them to the point of mass of the
        nearest cluster.
        :param k: amount of classes
        :param train_data: data used for training
        :param norm: used for determining the distance between points
        :param method: used method for minimize solver. The Nelder-Mead method works just fine for most circumstances
        :param cb0: initial guess for the initial cluster points. this may drasticly improve the results because if unset
        the classifier will pick code book at random
        """
        # assign values
        self.k = k
        self.train_data = train_data
        self.norm = norm
        self.method = method

        # get init code_book
        self.init_code_book = cb0 if cb0 is not None else self.__generate_init_code_book(k, train_data.shape[1],
                                                                                         train_data.min(),
                                                                                         tra_data.max())

        # self.maximize_expectation(max_iterations, epsilon, method)
        self.optimized_code_book, self.assignment_matrix = self.expectation_maximization(self.init_code_book,
                                                                                         self.train_data, epsilon,
                                                                                         max_iterations,
                                                                                         allow_code_book_pruning)

    def predict_cluster(self, data: np.ndarray):
        return self.__generate_assignment_table(self.optimized_code_book, data)

    def expectation_maximization(self, code_book: np.ndarray, train_data: np.ndarray, epsilon: float,
                                 max_iteration: int, allow_code_book_pruning: bool):
        iteration = 0
        assignment_matrix = None
        old_score = 0
        while iteration < max_iteration:
            assignment_matrix = self.__generate_assignment_table(code_book, train_data)
            code_book = self.__optimize_code_book(code_book, train_data, assignment_matrix, allow_code_book_pruning)
            score = self.__J(code_book, assignment_matrix, train_data)
            dif = abs(old_score - score)
            logging.debug(f"iteration {iteration}, score: {score}, old_score: {old_score} difference: {dif}")
            old_score = score
            if dif < epsilon:
                break
            iteration += 1
        return code_book, assignment_matrix

    @staticmethod
    def __generate_init_code_book(k, dim, min_val, max_val):
        array = np.zeros((k, dim))
        for k_i in range(k):
            array[k_i] += np.random.choice([min_val, max_val]) * np.random.random(dim)
        return array

    @staticmethod
    def __optimize_code_book(code_book: np.ndarray, train_data: np.ndarray, assignment_matrix: np.ndarray,
                             allow_code_book_pruning: bool):
        n_code_book = np.zeros(code_book.shape)
        remove_i = []
        for c_i, cb_ve in enumerate(n_code_book):
            as_m_i = assignment_matrix[:, c_i]
            as_m_i_sum = np.sum(as_m_i)
            if as_m_i_sum == 0:
                remove_i.append(c_i)
                continue
            cb_ve += np.sum((as_m_i * train_data.transpose()).transpose(), axis=0) / np.sum(as_m_i)

        if len(remove_i) > 0:
            logging.warning(f"unused code_book entry {remove_i}")
            if allow_code_book_pruning:
                n_code_book = np.delete(n_code_book, remove_i, axis=0)
                logging.debug(f"removed {remove_i}, new code book: {n_code_book}")
        return n_code_book

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

    def __J(self, code_book, assignment_matrix, train_data):
        J = 0
        for tv_i, train_vector in enumerate(train_data):
            for cv_i, cluster_vector in enumerate(code_book):
                J += assignment_matrix[tv_i, cv_i] * np.linalg.norm(train_vector - cluster_vector, ord=self.norm)
        return J


if __name__ == '__main__':
    # set loglevel
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    # init plot
    plt.style.use("dark_background")
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)

    # setup test data
    cl_se = np.array([[1, 1],
                      [1, 6],
                      [6, 6]])
    tra_data = generate_test_data(cl_se, 100, 1)

    # generate color dict
    labels = np.unique(list(range(cl_se.shape[0])))
    if len(labels) > len(mpl_colors.TABLEAU_COLORS.keys()):
        colors = np.random.choice(list(mpl_colors.XKCD_COLORS.keys()), size=len(labels))
    else:
        colors = np.random.choice(list(mpl_colors.TABLEAU_COLORS.keys()), size=len(labels))
    color_dict = dict(zip(labels, colors))
    legend_patches = []

    # init classifier
    classifier = KMeansClassifier(cl_se.shape[0], tra_data)

    cl_se_predicted = classifier.predict_cluster(cl_se)

    # generate test data
    # com = np.array(center_of_mass(classifier.optimized_code_book))
    com = np.mean(classifier.optimized_code_book, axis=0)
    test_data = generate_test_data(com, 50, 3)

    test_data_assigned = classifier.predict_cluster(test_data)
    legend_patches.append(ax0.plot(com[0],
                                   com[1],
                                   linestyle=" ",
                                   marker='+',
                                   color="white",
                                   label=f"center of mass")[0])

    for i in range(classifier.assignment_matrix.shape[1]):
        assigned_cluster = cl_se_predicted[:, i] == 1
        legend_patches.append(ax0.plot(cl_se[assigned_cluster, 0],
                                       cl_se[assigned_cluster, 1],
                                       linestyle=" ",
                                       marker='o',
                                       color=color_dict[labels[i]],
                                       label=f"cluster seed {labels[i]}")[0])

        legend_patches.append(ax0.plot(classifier.init_code_book[i, 0],
                                       classifier.init_code_book[i, 1],
                                       linestyle=" ",
                                       marker='^',
                                       color=color_dict[labels[i]],
                                       label=f"initial code book vector {labels[i]}")[0])

        legend_patches.append(ax0.plot(classifier.optimized_code_book[i, 0],
                                       classifier.optimized_code_book[i, 1],
                                       linestyle=" ",
                                       marker='x',
                                       color=color_dict[labels[i]],
                                       label=f"optimized code book vector {labels[i]}")[0])

        assigned_cluster = classifier.assignment_matrix[:, i] == 1
        legend_patches.append(ax0.plot(classifier.train_data[assigned_cluster, 0],
                                       classifier.train_data[assigned_cluster, 1],
                                       linestyle=" ",
                                       marker='.',
                                       color=color_dict[labels[i]],
                                       label=f"train data assigned to {labels[i]}")[0])

        assigned_cluster = test_data_assigned[:, i] == 1
        legend_patches.append(ax0.plot(test_data[assigned_cluster, 0],
                                       test_data[assigned_cluster, 1],
                                       linestyle=" ",
                                       marker='1',
                                       color=color_dict[labels[i]],
                                       label=f"test data assigned to {labels[i]}")[0])

    plt.legend(handles=legend_patches)
    plt.show()
