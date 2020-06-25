import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import logging
import os
import typing

from matplotlib import colors as mpl_colors


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


def __generate_test_data(cluster_seeds: np.ndarray, n, max_dif):
    dim = cluster_seeds.shape[1]
    array = np.zeros((n * cluster_seeds.shape[0], dim))
    index = 0
    for seed in cluster_seeds:
        for _i in range(n):
            while True:
                v = seed + np.random.choice([-1, 1], 2) * 2 * max_dif * np.random.uniform(size=dim)
                if np.linalg.norm(seed - v, ord=2) <= max_dif:
                    break
            array[index] += v
            index += 1
    return array


class KohonenNetworkClassifier:
    def __init__(self, a_neurons, train_data, max_generations,
                 learn_rate_k, neighbour_k,
                 learning_fall_off, neighbour_fall_off, norm=2, proto_type_spread=1000):
        # assigning attributes
        self.proto_type_spread = proto_type_spread
        self.learn_rate_k = learn_rate_k
        self.neighbour_k = neighbour_k
        self.neighbour_fall_off = neighbour_fall_off
        self.learning_fall_off = learning_fall_off
        self.norm = norm
        self.max_generations = max_generations
        self.train_data = train_data
        self.a_neurons = a_neurons
        self.neuron_map = self.__init_map(self.a_neurons)
        self.prototype_map = \
            self.__generate_prototype_tensor(a_neurons, train_data.shape[-1], self.neuron_map, lower=-proto_type_spread,
                                             upper=proto_type_spread)
        self.net_history = [[self.graph_friendly_network, self.active_neurons]]
        self.train()

    def train(self):
        for gen in range(self.max_generations):
            lamb = np.power(self.learning_fall_off, gen) * self.learn_rate_k
            sig = np.power(self.neighbour_fall_off, gen) * self.neighbour_k
            logging.debug(f"gen: {gen + 1}: {lamb}, {sig}")
            for t_v in self.train_data:
                a = self.__active_neurons(self.prototype_map, self.neuron_map)
                diff = np.linalg.norm(a - t_v, ord=self.norm, axis=1)
                j_star = self.neuron_map[diff.argsort()[0]]
                self.prototype_map[tuple(j_star)] += lamb * (t_v - self.prototype_map[tuple(j_star)])
                neighbours = self.__get_direct_neighbours(self.neuron_map, j_star, self.norm)
                for n in neighbours:
                    n_j_star = np.exp(-np.linalg.norm(n - j_star, self.norm) / sig)
                    self.prototype_map[tuple(n)] += lamb * n_j_star * (t_v - self.prototype_map[tuple(j_star)])
            self.net_history.append([self.graph_friendly_network, self.active_neurons])

    def assign_to_neuron(self, data, neurons_vectors=None):
        """

        :param data:
        :param neurons_vectors:
        :return:
        """
        p = self.active_neurons if neurons_vectors is None else neurons_vectors
        indices = np.zeros(data.shape[0])
        for i, d in enumerate(data):
            indices[i] += np.linalg.norm(p - d, ord=self.norm, axis=1).argsort()[0]
        return indices

    @property
    def active_neurons(self):
        return self.__active_neurons(self.prototype_map, self.neuron_map)

    @staticmethod
    def __active_neurons(prototypes, neuron_map):
        return np.array([prototypes[tuple(neuron)] for neuron in neuron_map])

    @property
    def graph_friendly_network(self):
        lines = []
        for i in range(self.neuron_map[:, 1].max() + 1):
            v = np.array([self.prototype_map[tuple(neuron)] for neuron in self.neuron_map[self.neuron_map[:, 1] == i]])
            h = np.array([self.prototype_map[tuple(neuron)] for neuron in self.neuron_map[self.neuron_map[:, 0] == i]])
            if h.shape[0] > 0:
                lines.append(h)
            if v.shape[0] > 0:
                lines.append(v)
        return lines

    @staticmethod
    def __init_map(k):
        a = np.zeros((k, 2), dtype=int)
        lines = np.floor(np.sqrt(k))
        columns = np.ceil(np.sqrt(k))
        a_index = 0
        for l_i in np.arange(lines, dtype=int):
            for c_i in np.arange(columns, dtype=int):
                if a_index >= a.shape[0]:
                    return a
                a[a_index] += np.array([l_i, c_i])
                a_index += 1
        return a

    @staticmethod
    def __generate_prototype_tensor(k, m, neuron_map, upper, lower):
        shape = (neuron_map[:, 0].max() + 1, neuron_map[:, 1].max() + 1, m)
        w = np.random.randint(lower, upper, shape) + np.random.choice([-1, 1], 2) * np.random.random(shape)
        return w

    @staticmethod
    def __get_direct_neighbours(neuron_map, neuron, norm=2):
        diff = np.linalg.norm(neuron_map - neuron, ord=norm, axis=1)

        is_edge = neuron[0] == neuron_map[:, 0].min() or neuron[0] == neuron_map[:, 0].max() or \
                  neuron[1] == neuron_map[:, 1].min() or neuron[1] == neuron_map[:, 1].max()

        is_corner = neuron[0] == neuron_map[:, 0].min() and neuron[1] == neuron_map[:, 1].min() or \
                    neuron[0] == neuron_map[:, 0].max() and neuron[1] == neuron_map[:, 1].max() or \
                    neuron[0] == neuron_map[:, 0].min() and neuron[1] == neuron_map[:, 1].max() or \
                    neuron[0] == neuron_map[:, 0].max() and neuron[1] == neuron_map[:, 1].min()

        if is_edge:
            if is_corner:
                # return indices of neighbours,
                return np.take(neuron_map, diff.argsort()[1:3], axis=0)
            return np.take(neuron_map, diff.argsort()[1:4], axis=0)
        return np.take(neuron_map, diff.argsort()[1:5], axis=0)


if __name__ == '__main__':

    def update_plot(frame, c, ax, drawings, tr_data, test_data, cluster_color_dict, network_color="blue", network_marker="x", network_line_style="-"):
        networks = c.net_history
        network_plot = networks[frame][0]
        neuron_vectors = networks[frame][1]
        # removing all old network lines
        while len(drawings) > 0:
            drawings.pop(0)
        # plotting updated ones
        for i, n in enumerate(network_plot):
            drawings.append(ax.plot(n[:, 0], n[:, 1],
                                    color=network_color, marker=network_marker, linestyle=network_line_style,
                                    label="network")[0])

        tr_indices = c.assign_to_neuron(tr_data, neuron_vectors)
        te_indices = c.assign_to_neuron(test_data, neuron_vectors)
        for i, v in enumerate(neuron_vectors):
            v_tr_data = train_data[tr_indices == i]
            v_te_data = test_data[te_indices == i]
            if v_tr_data.shape[0] > 0:
                drawings.append(ax.plot(v_tr_data[:, 0], v_tr_data[:, 1], color=cluster_color_dict[i], linestyle=" ",
                                        marker=".", label=f"training data assigned to {i}")[0])
            if train_data.shape[0] > 0:
                drawings.append(ax.plot(v_te_data[:, 0], v_te_data[:, 1], color=cluster_color_dict[i],
                                        linestyle=" ", marker="1", label=f"test data assigned to {i}")[0])
            drawings.append(ax.plot(v[0], v[1], color=cluster_color_dict[i], linestyle=" ", marker="+",
                                    label=f"network_prototype {i}")[0])
            plt.legend(handles=drawings[len(network_plot)-1:])
        return drawings

    #####################################################################################
    # interesting parameter to play with
    #####################################################################################
    #####################################################################################
    # trainings data settings

    # train data origin origin
    train_d_origin_spawn = np.array([[10, 10]])

    # amount origin origin descendants
    train_d_origins = 9

    # radius to spawn in train data origins around origin origin
    train_d_origin_diff = 2000

    # amout of train data points around origin
    train_d_points = 100

    # radius to spawn train data around data origins
    train_d_diff = train_d_origin_diff / 5

    # amout of test data points around origin
    test_d_points = 1000

    # radius to spawn train data around data origins
    test_d_diff = train_d_origin_diff

    #####################################################################################
    # classifier settings

    # the amount of neurons the network should use
    amount_neurons = train_d_origins

    # the maximum amount of trainings iterations allowed
    generation_maximum = 100

    learning_rate = 1
    learning_rate_dampening = 0.9

    neighbour_koef = .5
    neighbour_koef_dampening = 0.8

    prototype_spread = 1000

    #####################################################################################
    # boring stuff to test classifier and plot results
    #####################################################################################
    # set loglevel
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

    # set graph style
    plt.style.use("dark_background")
    labels = np.unique(list(range(amount_neurons)))
    if len(labels) > len(mpl_colors.TABLEAU_COLORS.keys()):
        colors = __get_unique_choice(list(mpl_colors.XKCD_COLORS.keys()), len(labels))
    else:
        colors = __get_unique_choice(list(mpl_colors.TABLEAU_COLORS.keys()), len(labels))
    color_dict = dict(zip(labels, colors))

    # init test data
    train_data_seeds = __generate_test_data(train_d_origin_spawn, train_d_origins, train_d_origin_diff)
    train_data = __generate_test_data(train_data_seeds, train_d_points, train_d_diff)
    test_data = __generate_test_data(train_d_origin_spawn, test_d_points, test_d_diff)
    # init plot stuff
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)

    #
    plots = []
    # init classifier
    c = KohonenNetworkClassifier(amount_neurons, train_data, generation_maximum, learn_rate_k=learning_rate,
                                 neighbour_k=neighbour_koef, learning_fall_off=learning_rate_dampening,
                                 neighbour_fall_off=neighbour_koef_dampening, proto_type_spread=prototype_spread)
    ax0.plot()

    ani = mpl.animation.FuncAnimation(fig, lambda n: update_plot(n, c, ax0, plots, train_data, test_data, color_dict),
                                      interval=2 * generation_maximum,
                                      frames=len(c.net_history), blit=True)

    plt.show()
