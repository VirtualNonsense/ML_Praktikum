import numpy as np
import pandas as pd
import typing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import logging
import os

from matplotlib import colors as mpl_colors


# noinspection DuplicatedCode
def __get_unique_choice(samples, choose: int):
    if len(np.unique(samples)) < choose:
        raise ValueError("impossible task")
    if len(np.unique(samples)) == choose:
        return np.unique(samples)
    a = []
    for _i in range(choose):
        while True:
            c = np.random.choice(np.unique(samples), size=1)
            if c not in a:
                break
        a.append(c[0])
    return a


def generate_test_data(data: pd.DataFrame):
    data_labels = np.unique(tr_data.columns[1:])
    t = [[] for _ in data_labels]

    f = np.arange(-2, 2, 0.5)
    for data_index, data_row in data.iterrows():
        for i, data_label in enumerate(data_labels):
            salt = random.choice(f) * random.random()
            t[i].append(data_row[data_label] + salt)

    t = np.transpose(np.array(t))

    return pd.DataFrame(data=t, columns=data_labels)


class KNN:
    def __init__(self, n: int, data, norm=2):
        self.n = n
        self.training_data: pd.DataFrame = data
        self.training_data_columns = np.array(self.training_data.columns[1:])
        self.norm = norm

    def predict_labels(self, test_data: pd.DataFrame) -> typing.List[str]:
        """
        This method takes vectors and tries to figure out the correct labels
        by using the K nearest neighbours method
        :param test_data: should be a DataFrame containing the __same__ column labels as training data
        :return:
        """

        # checking whether labels match
        common_columns = np.array(test_data.columns)
        if not np.array_equal(common_columns, self.training_data_columns):
            logging.warning("column labels don't match")
            common_columns = np.intersect1d(common_columns, self.training_data.columns[1:])
            if common_columns.shape[0] == 0:
                logging.critical("no common features, knn analysis impossible")
                raise ValueError("feature labels don't match")

        predicted = []
        for ted_index, ted_row in test_data.iterrows():
            difference = []
            # generating difference value for each column
            for c in common_columns:
                difference.append(ted_row[c] - np.array([v for v in self.training_data[c]]))

            # calculating norm for each vector
            difference = np.linalg.norm(np.transpose(np.array(difference)), self.norm, axis=1)

            # receiving indices for n smallest distances
            index_n_nearest = difference.argsort()[:self.n]

            # counting occurrences of each label
            unique, count = np.unique(self.training_data[self.training_data.columns[0]][index_n_nearest],
                                      return_counts=True)
            # saving label with most occurrences
            predicted.append(unique[count.argsort()[-1]])

        return predicted


if __name__ == '__main__':

    # set loglevel
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    # set graph style
    plt.style.use("dark_background")

    # load testdata with known labels
    tr_data = pd.read_csv("fruit.csv")
    # generate plausible data with unknown labels
    te_data = generate_test_data(tr_data)

    # generate color dict
    labels = np.unique(tr_data.labels)
    colors = __get_unique_choice(list(mpl_colors.XKCD_COLORS.keys()), len(labels))
    color_dict = dict(zip(labels, colors))

    # plot known data
    fig, ax = plt.subplots(1, 1)
    legend_patches = [patches.Patch(color=color_dict[key], label=key) for key in color_dict.keys()]
    for label in labels:
        sub_table = tr_data[tr_data['labels'] == label]
        ax.plot(sub_table.weight, sub_table.height, marker='x', color=color_dict[label], linestyle='None')
    plt.legend(handles=legend_patches)

    # classify unknown data
    classifier = KNN(3, tr_data)
    labels = classifier.predict_labels(te_data)

    # plot newly classified data
    index: int
    for index, row in te_data.iterrows():
        ax.plot(row.weight, row.height, marker='o', color=color_dict[labels[index]], linestyle='None')

    plt.show()
