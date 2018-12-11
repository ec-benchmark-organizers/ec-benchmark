import matplotlib.pyplot as plt

def plot_sample(x, y, ax):
    """
    Plots the sample of metocean data.

    Parameters
    ----------
    x : ndarray of doubles,
        The sample's first environmental variable.
    y : ndarray of doubles,
        The sample's second environmental variable.
    ax : Axes,
        Axes of the figure where the scatter plot should be drawn.
    """
    ax.scatter(x, y, s=2, c='k', label='observation')


def plot_contour(x, y, return_period, x_label='X1', y_label='X2', sample=None):
    """
    Plots the environmental contour.

    The method expects the coordinates to be ordered by angle.

    Parameters
    ----------
    x : ndarray of doubles,
        The contour's coordinates in the x-direction.
    y : ndarray of doubles,
        The contour's coordiantes in the y-direction.
    return_period : float,
        The environmental contour's return period in years.
    x_label : str,
        Label for the x-axis.
    y_label : str,
        Label for the y-axis.
    sample : list of lists of floats,
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
        plot_sample(sample[0], sample[1], ax)
    ec_label = str(return_period) + ' year contour'
    ax.plot(xplot, yplot, c='b', label=ec_label)

    # Format the figure.
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
