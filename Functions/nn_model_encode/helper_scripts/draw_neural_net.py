#!/Users/Dinesh/anaconda/bin/python
# Created on May 23, 2017
# Copied from this link: https://gist.github.com/craffel/2d727968c3aaebd10359

import pdb
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import collections as col
import seaborn as sns


class Draw_NN(object):
    '''Draw a neural network cartoon using matplotilb.

    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])

    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''

    def __init__(self, outputDir, outFileName, sizes_layer1, sizes_layer2,
                 layer_sizes, W1, W2, labels, num_labels_to_plot=0, geneName="",
                 left=0.1, right=0.9, bottom=0.1, top=0.9, figsize=(20, 20)):

        self.outputDir = outputDir  # to save the plot
        self.outFileName = outFileName

        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

        self.sizes_layer1 = sizes_layer1
        self.sizes_layer2 = sizes_layer2
        self.max_size_in_layers12 = np.max(self.sizes_layer1.tolist() + self.sizes_layer2.tolist())
        self.layer_sizes = layer_sizes
        self.W1 = W1
        self.W2 = W2
        self.labels = labels  # labels for the input nodes only
        self.geneName = geneName

        self.num_labels_to_plot = num_labels_to_plot
        self.edge_color_def = "gray"  # default
        self.edge_color_pos = "salmon"
        self.edge_color_neg = "steelblue"
        self.figsize = figsize  # this should be a tuple

    def plot_nn(self):
        '''
            :parameters:
                - zorder : int
                    For the order that elements get plotted (higher is on top).
                    Info on z-order: http://matplotlib.org/examples/pylab_examples/zorder_demo.html
        '''
        fig = plt.figure(figsize=self.figsize)
        sns.set(font_scale=1)
        ax = fig.gca()
        ax.axis('off')

        v_spacing = (self.top - self.bottom) / float(max(self.layer_sizes))
        h_spacing = (self.right - self.left) / float(len(self.layer_sizes) - 1)

        # Nodes
        input_layer_sizes = []
        layer_top_forInputs = -1
        for n, layer_size in enumerate(self.layer_sizes):
            # compute the node sizes
            if (n == 0):
                W = self.W1
                sizes = self.sizes_layer1  # np.sum(W, axis=1) # add up the columns in W1 (of dimension (num_inputnodes, num_hiddenLayer1nodes), for eg.)
            elif (n == 1):
                W = self.W2
                sizes = self.sizes_layer2  # np.sum(self.W1, axis=0) + np.sum(W, axis=1) # add incoming weights (aggregating over rows) plus outgoing weights (aggregating over columns)
            else:
                W = self.W2  # assuming there is just one hidden layer !
                sizes = np.sum(W, axis=0)  # aggregating over rows

            tmp_sizes = [abs(x) for x in sizes]  # make the sizes positive in this temporary sizes vector
            sizes = 2 * (tmp_sizes / self.max_size_in_layers12)  # previously was being divided by np.max(tmp_sizes)
            if (n == 0):
                input_layer_sizes = sizes

            layer_top = v_spacing * (layer_size - 1) / 2. + (self.top + self.bottom) / 2.
            if (n == 0):  # save the layer_top for the input layer to use for labels later.
                layer_top_forInputs = layer_top

            for m in xrange(layer_size):
                circle = plt.Circle((n * h_spacing + self.left, layer_top - m * v_spacing), (v_spacing * sizes[m]) / 4, color='w', ec='k', zorder=4)
                ax.add_artist(circle)

        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (self.top + self.bottom) / 2.
            layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (self.top + self.bottom) / 2.

            if (n == 0):
                W = self.W1
            elif (n == 1):
                W = self.W2
            else:
                raise Exception("Currently only assuming there is one hidden ayer..")

            for m in xrange(layer_size_a):
                for o in xrange(layer_size_b):
                    if (W[m, o] < 0):
                        useColor = self.edge_color_neg  # was _neg before
                    else:
                        useColor = self.edge_color_pos  # was pos before
                    line = plt.Line2D([n * h_spacing + self.left, (n + 1) * h_spacing + self.left],
                                      [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c=useColor, alpha=0.5, lw=abs(W[m, o]))  # was lw=W[m, o] initially, but the plot saved had edge missing (but not in jupyter -- which was weird..); so changed to 0.35
                    ax.add_artist(line)

        # Add input layer labels
        if (self.num_labels_to_plot > 0):
            dict_labels = col.OrderedDict(zip(self.labels, input_layer_sizes))
            list_labels_sorted = sorted(dict_labels.items(), reverse=True, key=lambda x: x[1])  # largest size at the beginning of the dict

            labels_to_plot = [x[0] for x in list_labels_sorted]  # i.e. get first elements in this list of tuples of form (label, size)
            labels_plotted = 0

            for i in range(0, len(input_layer_sizes)):  # going through all input nodes
                if (self.num_labels_to_plot > labels_to_plot) and (labels_plotted == len(labels_to_plot)):  # if asking to plot more than what can be
                    break
                if (labels_plotted == self.num_labels_to_plot):
                    break
                if (labels_to_plot.__contains__(self.labels[i])):
                    label_to_plot, size = list_labels_sorted[labels_plotted]  # this is our current index of label to plot

                    text_x = self.left
                    text_y = layer_top_forInputs - (self.labels.index(label_to_plot) * v_spacing)

                    plt.annotate(label_to_plot, xy=(text_x, text_y), xytext=(-10, 10), textcoords="offset points", ha='right', va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.4', fc='lightblue', alpha=0.05),
                                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'), zorder=10)

                    labels_plotted += 1

        if (self.geneName != ""):
            plt.title("Model for " + self.geneName, fontsize=20)

        plt.tight_layout()
        plt.subplots_adjust(left=0.28, right=0.9, top=0.8, bottom=0.15)  # else the labels can be chopped off
        plt.savefig(self.outputDir + "/" + self.outFileName)
        plt.close()
        return fig


if __name__ == "__main__":
    layer_sizes = [3, 3, 1]
    w1 = np.array([[1, 2, 3], [0.5, 1, 3], [3, 3, 3]])  # 2 by 3
    w2 = np.array([[3], [3], [3]])  # 3 by 1
    labels = ["chr1:adasfasdf-adsgadsgdsh", "chr1:bagdadsg-gadsasha", "chr1:cgashsdh-hasdhsha"]  # labels for the input nodes
    d = Draw_NN(layer_sizes, w1, w2, labels, num_labels_to_plot=3, figsize=(8, 6))
    d.plot_nn()
    plt.close()
