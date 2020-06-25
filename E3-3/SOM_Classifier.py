import numpy as np
import matplotlib.pyplot as plt
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
                 learn_rate=2, sigma=2,
                 mu=0.5, nu=0.5,
                 p_func=None, norm=2):
        # assigning attributes
        self.l_lambda = learn_rate
        self.sigma = sigma
        self.nu = nu
        self.mu = mu
        self.norm = norm
        self.max_generations = max_generations
        self.train_data = train_data
        self.a_neurons = a_neurons
        self.neuron_map = self.__init_map(self.a_neurons)
        self.prototype_map = self.__generate_prototype_tensor(a_neurons, train_data.shape[-1], self.neuron_map)
        self.__p_func = p_func
        self.__plot_network()
        self.train()

    def train(self):

        for gen in range(self.max_generations):
            logging.debug(f"gen: {gen+1}")
            for t_v in self.train_data:
                j_star = self.neuron_map[np.linalg.norm(self.active_neurons - t_v, ord=self.norm).argsort()[0]]
                lamb = np.power(self.mu, gen) * self.l_lambda
                self.prototype_map[tuple(j_star)] += lamb * (t_v - self.prototype_map[tuple(j_star)])
                neighbours = self.__get_direct_neighbours(self.neuron_map, j_star, self.norm)
                for n in neighbours:
                    sig = np.power(self.nu, gen) * self.sigma
                    n_j_star = np.exp(-np.linalg.norm(n-j_star, self.norm) / sig)
                    self.prototype_map[tuple(n)] += lamb * n_j_star * (t_v - self.prototype_map[tuple(j_star)])
            # self.__plot_network()

    @property
    def active_neurons(self):
        return np.array([self.prototype_map[tuple(neuron)] for neuron in self.neuron_map])

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

    def __plot_network(self):
        if self.__p_func is not None:
            self.__p_func(self.graph_friendly_network)

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
    def __generate_prototype_tensor(k, m, neuron_map, upper=-1000, lower=1000):
        shape = (neuron_map[:, 0].max() + 1, neuron_map[:, 1].max() + 1, m)
        w = np.random.randint(upper, lower, shape) + np.random.choice([-1, 1], 2) * np.random.random(shape)
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

    def update_plot(network, fig, ax, lines, color="blue", marker="x", linestyle="-"):
        # removing all old network lines
        while len(lines) > 0:
            lines.pop(0)
        # plotting updated ones
        for i, n in enumerate(network):
            lines.append(ax.plot(n[:, 0], n[:, 1], color=color, marker=marker, linestyle=linestyle)[0])
        fig.canvas.draw()
        return


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
    train_d_origin_diff = 3

    # amout of train data points around origin
    train_d_points = 10

    # radius to spawn train data around data origins
    train_d_diff = train_d_origin_diff / 4

    #####################################################################################
    # classifier settings

    # the amount of neurons the network should use
    amount_neurons = train_d_origins

    # the maximum amount of trainings iterations allowed
    generation_maximum = 10

    # function used for neuron displacement
    learning_rate = lambda x: 1 / x

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
    c = KohonenNetworkClassifier(amount_neurons, train_data, generation_maximum,
                                 p_func=lambda network: update_plot(network, fig, ax0, network_lines))

    print(network_lines)
    plt.show()