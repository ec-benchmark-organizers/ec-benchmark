import numpy as np
import matplotlib.pyplot as plt


def plot_sample(x, y, ax, x_inside=None, y_inside=None ,x_outside=None,
                y_outside=None, return_period=None,
                do_plot_extreme=[True, True, True, True]):
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
    return_period : int, optional
        Return period in years. Is used in legend for describing the inside and
        outside datapoints.
    do_plot_extreme : ndarray of booleans, optional
        Specifies which extremes should be plotted.
        The order is [min(x), max(x), min(y), max(y)].
    """
    if x_inside is not None and y_inside is not None:
        if return_period:
            inside_label = 'inside ' + str(return_period) + '-yr contour'
            outside_label = 'outside ' + str(return_period) + '-yr contour'
        else:
            inside_label = 'inside contour'
            outside_label = 'outside contour'
        ax.scatter(x_inside, y_inside, s=11, alpha=0.5, c='k', marker='o',
                   label=inside_label)
        ax.scatter(x_outside, y_outside, s=9, alpha=0.5, c='r', marker='D',
                   label=outside_label)
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
    if any(d == True for d in do_plot_extreme):
        ax.scatter(x_extremes, y_extremes,  s=40, c='g', marker='*',
                   label='observed extreme')


def plot_contour(x, y, return_period, ax, x_label='', y_label='',
                 line_style='b-', sample=None):
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
    ax : Axes
        Axes of the figure where the contour should be plotted.
    x_label : str, optional
        Label for the x-axis.
    y_label : str, optional
        Label for the y-axis.
    line_style : str, optional
        Matplotlib line style.
    sample : list of lists of floats, optional
        Sample of environmental states and optional additional information:
        Element 0:
            x : ndarray of doubles
                The sample's first environmental variable.
        Element 1:
            y : ndarray of doubles
                The sample's second environmental variable.
        Element 2, optional:
                x_inside : ndarray of doubles
                Values in the first dimension of the points inside the contour.
        Element 3, optional:
            y_inside : ndarray of doubles
            Values in the second dimension of the points inside the contour.
        Element 4, optional:
            x_outside : ndarray of doubles
            Values in the first dimension of the points outside the contour.
        Element 5, optional:
            y_outside : ndarray of doubles
                Values in the second dimension of the points outside the contour.
        Element 6, optional:
            return_period_20 : int, optional
                Return period in years. Is used in legend for describing the inside and
                outside datapoints.
        Element 7, optional:
            do_plot_extreme : ndarray of booleans
                Specifies which extremes should be plotted.
                The order is [min(x), max(x), min(y), max(y)].
    """
    # For generating a closed contour: add the first coordinate at the end.
    xplot = x.tolist()
    xplot.append(x[0])
    yplot = y.tolist()
    yplot.append(y[0])

    # Plot the contour and, if provided, also the sample.
    if sample:
        if len(sample) > 2:
            plot_sample(x=sample[0],
                        y=sample[1],
                        ax=ax,
                        x_inside=sample[2],
                        y_inside=sample[3],
                        x_outside=sample[4],
                        y_outside=sample[5],
                        return_period=sample[6],
                        do_plot_extreme=sample[7])
        else:
            plot_sample(x=sample[0], y=sample[1], ax=ax)
    ec_label = str(return_period) + '-yr contour'
    ax.plot(xplot, yplot, line_style, label=ec_label)

    # Format the figure.
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    y_lim_factor = 1.2
    if sample:
        # If there is not enough space for the legend in the upper left corner:
        # make space for it.
        max_index = np.where(sample[1] == max(sample[1]))
        if sample[0][max_index] < 0.6 * max(max(x), max(sample[0])):
            y_lim_factor = 1.35

        upper_ylim = max(max(y), max(sample[1])) * y_lim_factor
    else:
        upper_ylim = max(y) * y_lim_factor
    plt.ylim((0, upper_ylim))

    # Remove axis on the right and on the top (Matlab 'box off').
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
