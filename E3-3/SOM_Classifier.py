import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation
import logging
import os
import typing


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
        self.net_history = [self.graph_friendly_network]
        self.train()

    def train(self):

        for gen in range(self.max_generations):
            lamb = np.power(self.learning_fall_off, gen) * self.learn_rate_k
            sig = np.power(self.neighbour_fall_off, gen) * self.neighbour_k
            logging.debug(f"gen: {gen + 1}: {lamb}, {sig}")
            for t_v in self.train_data:
                a = self.active_neurons(self.prototype_map, self.neuron_map)
                diff = np.linalg.norm(a - t_v, ord=self.norm, axis=1)
                j_star = self.neuron_map[diff.argsort()[0]]
                self.prototype_map[tuple(j_star)] += lamb * (t_v - self.prototype_map[tuple(j_star)])
                neighbours = self.__get_direct_neighbours(self.neuron_map, j_star, self.norm)
                for n in neighbours:
                    n_j_star = np.exp(-np.linalg.norm(n - j_star, self.norm) / sig)
                    self.prototype_map[tuple(n)] += lamb * n_j_star * (t_v - self.prototype_map[tuple(j_star)])
            self.net_history.append(self.graph_friendly_network)

    @staticmethod
    def active_neurons(prototypes, neuron_map):
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

    def update_plot(frame, network, ax, lines, color="blue", marker="x", linestyle="-"):
        # removing all old network lines
        network = network[frame]
        while len(lines) > 0:
            lines.pop(0)
        # plotting updated ones
        for i, n in enumerate(network):
            lines.append(ax.plot(n[:, 0], n[:, 1], color=color, marker=marker, linestyle=linestyle)[0])
        return lines


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

    # init test data
    train_data_seeds = __generate_test_data(train_d_origin_spawn, train_d_origins, train_d_origin_diff)
    train_data = __generate_test_data(train_data_seeds, train_d_points, train_d_diff)

    # init plot stuff
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 1, 1)

    #
    network_lines = []
    # init classifier
    c = KohonenNetworkClassifier(amount_neurons, train_data, generation_maximum, learn_rate_k=learning_rate,
                                 neighbour_k=neighbour_koef, learning_fall_off=learning_rate_dampening,
                                 neighbour_fall_off=neighbour_koef_dampening, proto_type_spread=prototype_spread)
    ax0.plot()

    ax0.plot(c.train_data[:, 0],
             c.train_data[:, 1],
             linestyle=" ",
             marker='.',
             color="red",
             label=f"train data assigned to")

    ani = mpl.animation.FuncAnimation(fig, lambda n: update_plot(n, c.net_history, ax0, network_lines),
                                      interval=2 * generation_maximum,
                                      frames=len(c.net_history), blit=True)

    plt.show()
