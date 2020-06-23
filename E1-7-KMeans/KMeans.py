import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import typing

from matplotlib import colors as mpl_colors
from scipy.optimize import minimize


def generate_test_data(cluster_seeds: np.ndarray, n, max_dif):
    dim = cluster_seeds.shape[1]
    array = np.zeros((n * cluster_seeds.shape[0], dim))
    index = 0
    for seed in cluster_seeds:
        for i in range(n):
            while True:
                v = seed + np.random.choice([-1, 1], 2) * max_dif * np.random.uniform(size=dim)
                if np.linalg.norm(seed - v, ord=2) <= max_dif:
                    break
            array[index] += v
            index += 1

    return array


# noinspection PyArgumentList
class KMeansClassifier:
    """
    This classifier splits un
    """

    def __init__(self, k: int,
                 train_data: np.ndarray,
                 norm: int = 2,
                 method='Nelder-Mead',
                 cb0=None,
                 train_data_init_cb=False,
                 max_iterations=11,
                 epsilon=1e-3,
                 allow_code_book_pruning=False):
        """
        This classifier tries to split a given data set into k classes, by assigning them to the point of mass of the
        nearest cluster.
        :param k: amount of classes
        :param train_data: data used for training
        :param norm: [OPTIONAL] used for determining the distance between points
        :param method: [OPTIONAL] used method for minimize solver. The Nelder-Mead method works just fine for most
        circumstances
        :param cb0: [OPTIONAL] initial guess for the initial cluster points. this may drastically improve the results
        because if unset the classifier will pick code book at random
        :param train_data_init_cb: [OPTIONAL] if set to True (cb0 is none) random points from train data will be used as
        initial code_book (use when the samples sizes of each cluster alike)
        :param max_iterations: [OPTIONAL] the maximum amount of steps allowed for code_book optimisation
        :param epsilon: [OPTIONAL] this determines the maximum discrepancy between optimization scores
        :param allow_code_book_pruning: [OPTIONAL] I noticed that in some cases a code_book vector remains unassigned
        When set to true these kind of vectors will be removed.
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
        self.init_code_book = cb0 if cb0 is not None \
            else train_data[np.random.choice(tra_data.shape[0], k), :] if train_data_init_cb \
            else self.__generate_init_code_book(k, train_data.shape[1], train_data.min(), tra_data.max())

        # Optimize code book
        self.optimized_code_book, self.assignment_matrix, self.code_book_history = self.expectation_maximization(
            self.init_code_book,
            self.train_data, epsilon,
            max_iterations,
            allow_code_book_pruning)

    def predict_cluster(self, data: np.ndarray, code_book=None):
        """
        Predict unlabeled data based on a pre trained code_book
        :param data: data to classify
        :param code_book: [OPTIONAL] maybe useful to animate convergence
        :return:
        """
        if code_book is not None:
            return self.__generate_assignment_matrix(code_book, data)
        return self.__generate_assignment_matrix(self.optimized_code_book, data)

    def expectation_maximization(self, code_book: np.ndarray, train_data: np.ndarray, epsilon: float,
                                 max_iteration: int, allow_code_book_pruning: bool = False) -> typing.Tuple[
        np.ndarray, np.ndarray, typing.List[np.ndarray]]:
        """
        Function optimise the code book.
        :param code_book: Initial guesses for the code_book
        :param train_data: Data used for code book optimisation
        :param epsilon: this determines the maximum discrepancy between optimization scores
        :param max_iteration: the maximum amount of steps allowed for code_book optimisation
        :param allow_code_book_pruning: [OPTIONAL] I noticed that in some cases a code_book vector remains unassigned
        When set to true these kind of vectors will be removed.
        :return: (code_book, assignment_matrix, code_book_history)
        """
        # initialize variables
        iteration = 0
        assignment_matrix = None
        old_score = 0
        code_book_history = [code_book]

        # start iteration
        while iteration < max_iteration:
            # generate assignment matrix to current code_book
            assignment_matrix = self.__generate_assignment_matrix(code_book, train_data)
            # optimize adjust current code_book to assigned training data
            code_book = self.__optimize_code_book(code_book, train_data, assignment_matrix, allow_code_book_pruning)
            # get score of current parameter
            score = self.__J(code_book, assignment_matrix, train_data)
            dif = abs(old_score - score)
            logging.debug(f"iteration {iteration}, score: {score}, old_score: {old_score} difference: {dif}")
            old_score = score
            # hold if threshold is reached
            if dif < epsilon:
                break
            code_book_history.append(code_book)
            iteration += 1
        if iteration > max_iteration:
            logging.warning("max iteration has been exceeded ")
        return code_book, assignment_matrix, code_book_history

    @staticmethod
    def __generate_init_code_book(k, dim, min_val, max_val):
        """
        sloppy method to generate the initial code book
        :param k: amount code book vectors
        :param dim: amount of dimensions per vector
        :param min_val: upper limit
        :param max_val: lower limit
        :return:
        """
        array = np.zeros((k, dim))
        for k_i in range(k):
            array[k_i] += np.random.choice([-1, 1], 2) * np.random.choice([min_val, max_val], 2) * np.random.random(dim)
        return array

    @staticmethod
    def __optimize_code_book(code_book: np.ndarray, train_data: np.ndarray, assignment_matrix: np.ndarray,
                             allow_code_book_pruning: bool = False):
        """

        :param code_book:
        :param train_data:
        :param assignment_matrix:
        :param allow_code_book_pruning: [OPTIONAL] I noticed that in some cases a code_book vector remains unassigned
        When set to true these kind of vectors will be removed.
        :return:
        """
        n_code_book = np.zeros(code_book.shape)
        unused_cb_i = []
        for c_i, cb_ve in enumerate(n_code_book):
            as_m_i = assignment_matrix[:, c_i]
            as_m_i_sum = np.sum(as_m_i)
            if as_m_i_sum == 0:
                unused_cb_i.append(c_i)
                continue
            cb_ve += np.sum((as_m_i * train_data.transpose()).transpose(), axis=0) / np.sum(as_m_i)

        if len(unused_cb_i) > 0:
            logging.warning(f"unused code_book entry {unused_cb_i}")
            if allow_code_book_pruning:
                n_code_book = np.delete(n_code_book, unused_cb_i, axis=0)
                logging.debug(f"removed {unused_cb_i}, new code book: {n_code_book}")
        return n_code_book

    def __generate_assignment_matrix(self, cluster_matrix: np.ndarray, data_matrix: np.ndarray):
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
    com = np.mean(classifier.optimized_code_book, axis=0)
    # test_data = generate_test_data(com.reshape((1, 2)), 50, 3)
    test_data = generate_test_data(classifier.optimized_code_book, 100, 4)

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
