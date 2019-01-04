import numpy as np
import matplotlib.pyplot as plt


def plot_sample(x, y, ax, x_inside=None, y_inside=None ,x_outside=None,
                y_outside=None, do_plot_extreme=[True, True, True, True]):
    """
    Plots the sample of metocean data.

    Parameters
    ----------
    x : ndarray of doubles
        The sample's first environmental variable.
    y : ndarray of doubles
        The sample's second environmental variable.
    ax : Axes
        Axes of the figure where the scatter plot should be drawn.
    x_inside : ndarray of doubles
        Values in the first dimension of the points inside the contour.
    y_inside : ndarray of doubles
        Values in the second dimension of the points inside the contour.
    x_outside : ndarray of doubles
        Values in the first dimension of the points outside the contour.
    y_outside : ndarray of doubles
        Values in the second dimension of the points outside the contour.
    do_plot_extreme : ndarray of booleans
        Specifies which extremes should be plotted.
        The order is [min(x), max(x), min(y), max(y)].
    """
    if x_inside is not None and y_inside is not None:
        ax.scatter(x_inside, y_inside, s=11, alpha=0.5, c='k', marker='o',
                   label='inside contour')
        ax.scatter(x_outside, y_outside, s=9, alpha=0.5, c='r', marker='D',
                   label='outside contour')
    else:
        ax.scatter(x, y, s=40, alpha=0.5, c='k', marker='.', label='observation')
    x_extremes = np.empty((4,1,))
    x_extremes[:] = np.nan
    y_extremes = np.empty((4,1,))
    y_extremes[:] = np.nan
    if do_plot_extreme[0]:
        index_min_x = np.argmin(x)
        x_extremes[0] = x[index_min_x]
        y_extremes[0] = y[index_min_x]
    if do_plot_extreme[1]:
        index_max_x = np.argmax(x)
        x_extremes[1] = x[index_max_x]
        y_extremes[1] = y[index_max_x]
    if do_plot_extreme[2]:
        index_min_y = np.argmin(y)
        x_extremes[2] = x[index_min_y]
        y_extremes[2] = y[index_min_y]
    if do_plot_extreme[3]:
        index_max_y = np.argmax(y)
        x_extremes[3] = x[index_max_y]
        y_extremes[3] = y[index_max_y]
    ax.scatter(x_extremes, y_extremes,  s=40, c='g', marker='*', label='observed extreme')


def plot_contour(x, y, return_period, x_label='X1', y_label='X2', sample=None):
    """
    Plots the environmental contour.

    The method expects the coordinates to be ordered by angle.

    Parameters
    ----------
    x : ndarray of doubles
        The contour's coordinates in the x-direction.
    y : ndarray of doubles
        The contour's coordiantes in the y-direction.
    return_period : float
        The environmental contour's return period in years.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    sample : list of lists of floats
        Sample of environmental states.
    """
    # For generating a closed contour: add the first coordinate at the end.
    xplot = x.tolist()
    xplot.append(x[0])
    yplot = y.tolist()
    yplot.append(y[0])

    # Plot the contour and, if provided, also the sample.
    fig = plt.figure(figsize=(5,5), dpi=150)
    ax = fig.add_subplot(111)
    if sample:
        if len(sample) > 2:
            plot_sample(x=sample[0],
                        y=sample[1],
                        ax=ax,
                        x_inside=sample[2],
                        y_inside=sample[3],
                        x_outside=sample[4],
                        y_outside=sample[5],
                        do_plot_extreme=sample[6])
        else:
            plot_sample(x=sample[0], y=sample[1], ax=ax)
    ec_label = str(return_period) + ' year contour'
    ax.plot(xplot, yplot, c='b', label=ec_label)

    # Format the figure.
    if x_label == 'zero-up-crossing period (s)':
        plt.legend(loc='upper right', frameon=False)
    else:
        plt.legend(loc='upper left', frameon=False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if sample:
        upper_ylim = max(max(y), max(sample[1])) * 1.2
    else:
        upper_ylim = max(y) * 1.2
    plt.ylim((0, upper_ylim))

    # Remove axis on the right and on the top (Matlab 'box off').
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    plt.show()
