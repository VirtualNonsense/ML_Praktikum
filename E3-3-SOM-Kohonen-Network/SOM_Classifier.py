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
    def __init__(self, a_neurons: int, train_data: np.ndarray, max_generations: int,
                 learn_rate_function, neighbour_function, norm: int = 2, proto_type_spread: float = 1000, epsilon=0.):
        """
        :param a_neurons: amount of neurons
        :param train_data: input data for training
        :param max_generations: amount of training iterations
        :param learn_rate_function: f(generation): N_0 -> R. The return of this function scales the amount of change
        applied in a given generation. If it falls over time/generation the network will harden
        :param neighbour_function:  f(generation): N_0 -> R. The return of this function scales the amount of change
        applied to the neighbours of node in a given generation.
        :param norm: [OPTIONAl] sets the way distances are calculated e.g. 2 == euclidean distance
        :param proto_type_spread: [OPTIONAL] side length of a square, with zero in it's center in which the prototypes
        spawn in
        """
        # assigning attributes
        self.proto_type_spread = proto_type_spread
        self.learn_rate_f = learn_rate_function
        self.neighbour_f = neighbour_function
        self.norm = norm
        self.max_generations = max_generations
        self.train_data = train_data
        self.a_neurons = a_neurons
        self.epsilon = epsilon
        self.neuron_map = self.__init_map(self.a_neurons)
        self.prototype_map = \
            self.__generate_prototype_tensor(a_neurons, train_data.shape[-1], self.neuron_map, lower=-proto_type_spread,
                                             upper=proto_type_spread, t_mean=np.mean(train_data, axis=0))
        self.net_history = [[self.graph_friendly_network, self.active_neurons]]
        self.train()

    def train(self):
        for gen in range(self.max_generations):
            # calc learning koef
            lamb = self.learn_rate_f(gen)

            # calc neighbour koef
            sig = self.neighbour_f(gen)
            logging.debug(f"gen: {gen + 1}: {lamb}, {sig}")
            for t_v in self.train_data:
                # get nearest neuron
                a = self.__active_neurons(self.prototype_map, self.neuron_map)
                diff = np.linalg.norm(a - t_v, ord=self.norm, axis=1)
                j_star = self.neuron_map[diff.argsort()[0]]

                # calc new prototype
                delta = lamb * (t_v - self.prototype_map[tuple(j_star)])
                if np.linalg.norm(delta, ord=2) < self.epsilon:
                    logging.debug(f'{np.linalg.norm(delta, ord=2)} < {self.epsilon}: training complete.')
                    return

                self.prototype_map[tuple(j_star)] += delta
                # get neighbours
                neighbours = self.__get_direct_neighbours(self.neuron_map, j_star, self.norm)
                for n in neighbours:
                    # calc new neighbour prototype
                    n_j_star = np.exp(-np.linalg.norm(n - j_star, self.norm) / sig)
                    self.prototype_map[tuple(n)] += lamb * n_j_star * (t_v - self.prototype_map[tuple(j_star)])

            # add current state to history (for plots only)
            self.net_history.append([self.graph_friendly_network, self.active_neurons])

    def assign_to_neuron(self, data, neurons_vectors=None):
        """
        indices to assign data to active neurons
        :param data:
        :param neurons_vectors: eg. an old state of active_neurons (interesting for plotting)
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
        """
        returns one line/ndarray for each row and column
        :return:
        """
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
        lines = np.ceil(np.sqrt(k))
        columns = np.ceil(np.sqrt(k))
        a_index = 0
        # lines/columns + 1 np.arange interprets this parameters as size otherwise
        for l_i in np.arange(lines, dtype=int):
            for c_i in np.arange(columns, dtype=int):
                if a_index >= a.shape[0]:
                    return a
                a[a_index] += np.array([l_i, c_i])
                a_index += 1
        return a

    @staticmethod
    def __generate_prototype_tensor(k, m, neuron_map, upper, lower, t_mean):
        shape = (neuron_map[:, 0].max() + 1, neuron_map[:, 1].max() + 1, m)
        w = t_mean + np.random.randint(lower, upper, shape) + np.random.choice([-1, 1], 2) * np.random.random(shape)
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
                # return indices of neighbours, ignore the first pos, because it's neuron
                return np.take(neuron_map, diff.argsort()[1:3], axis=0)
            return np.take(neuron_map, diff.argsort()[1:4], axis=0)
        return np.take(neuron_map, diff.argsort()[1:5], axis=0)


if __name__ == '__main__':

    def update_plot(frame, c, ax, ax_title, drawings, tr_data, test_data, cluster_color_dict, network_color="blue",
                    network_marker="x", network_line_style="-"):
        networks = c.net_history
        network_plot = networks[frame][0]
        neuron_vectors = networks[frame][1]
        ax_title.set_text(f"Generation {frame + 1}")
        # removing all old network lines
        # it would be more elegant to set the changed data in the existing plots
        # but it seems to be a bit buggy as well and i don't have much time left for my other stuff :/
        while len(drawings) > 0:
            drawings.pop(0)

        # clear ax after each cycle otherwise there will be a ghost image of the final network
        # for some reason it remains even if i change the data of each line in drawings manually
        # instead of just deleting each drawing... matplotlib is weird....
        if frame == 0:
            ax.clear()
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

        plt.legend(handles=drawings[len(network_plot) - 1:], bbox_to_anchor=(1.01, 1))
        plt.subplots_adjust(left=0.1, right=.6)
        return drawings + [title]


    #####################################################################################
    # interesting parameter to play with
    #####################################################################################
    #####################################################################################
    # trainings data settings

    # train data origin origin
    train_d_origin_spawn = np.array([[10, 10]])

    # amount origin origin descendants
    train_d_origins = 10

    # radius to spawn in train data origins around origin origin
    train_d_origin_diff = 20

    # amout of train data points around origin
    train_d_points = 100

    # radius to spawn train data around data origins
    train_d_diff = train_d_origin_diff / 6

    # amount of test data points around origin
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
    learning_rate_dampening = 0.5
    lf = lambda gen: np.exp(-learning_rate_dampening*gen) * learning_rate

    neighbour_koef = 6
    neighbour_koef_dampening = .6
    nf = lambda gen: np.exp(-neighbour_koef_dampening * gen) * neighbour_koef

    prototype_spread = train_d_origin_diff / 2

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
    c = KohonenNetworkClassifier(amount_neurons, train_data, generation_maximum, learn_rate_function=lf,
                                 neighbour_function=nf, proto_type_spread=prototype_spread, epsilon=1e-3)

    title = ax0.text(0.05, 0.92, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                     transform=ax0.transAxes, ha="left")

    ani = mpl.animation.FuncAnimation(fig,
                                      lambda n: update_plot(n, c, ax0, title, plots, train_data, test_data, color_dict),
                                      interval=300,
                                      frames=len(c.net_history),
                                      blit=True)

    plt.show()
