from plot_generic import alpha_shape, convert_ndarray_list_to_multipoint
from descartes import PolygonPatch
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
    ax :
    """
    ax.scatter(x, y, s=2, c='k', label='observation')


def plot_contour(x, y, return_period, x_label='X1', y_label='X2', sample=None):
    """
    Plots the environmental contour.

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

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if sample:
        plot_sample(sample[0], sample[1], ax)
    ax.scatter(x, y, s=15, c='b',
               label='design condition')
    try:
        alpha = .3
        concave_hull, edge_points = alpha_shape(
            convert_ndarray_list_to_multipoint([x, y]),
            alpha=alpha)
        patch_design_region = PolygonPatch(
            concave_hull, fc='#999999', linestyle='None', fill=True,
            zorder=-2, label='design region')
        ec_label = str(return_period) + ' year environmental contour'
        patch_environmental_contour = PolygonPatch(
            concave_hull, ec='b', fill=False, zorder=-1,
            label=ec_label)
        ax.add_patch(patch_design_region)
        ax.add_patch(patch_environmental_contour)
    except(ZeroDivisionError):  # alpha_shape() can throw these
        print('Encountered a ZeroDivisionError when using alpha_shape.'
              'Consequently no contour is plotted.')

    plt.legend(loc='lower right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()
