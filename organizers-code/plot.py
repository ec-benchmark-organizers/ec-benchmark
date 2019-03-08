import numpy as np
import matplotlib.pyplot as plt


def plot_sample(plotted_sample):
    """
    Plots the sample of metocean data.

    Parameters
    ----------
    plotted_sample : PlottedSample,
        The sample that should be plotted and its meta information.
    """
    ps = plotted_sample
    x = ps.x
    y = ps.y
    if ps.x_inside is not None and ps.y_inside is not None:
        if ps.return_period:
            inside_label = 'inside ' + str(ps.return_period) + '-yr contour'
            outside_label = 'outside ' + str(ps.return_period) + '-yr contour'
        else:
            inside_label = 'inside contour'
            outside_label = 'outside contour'
        ps.ax.scatter(ps.x_inside, ps.y_inside, s=11, alpha=0.5, c='k',
                      marker='o', label=inside_label)
        ps.ax.scatter(ps.x_outside, ps.y_outside, s=9, alpha=0.5, c='r',
                      marker='D', label=outside_label)
    else:
        if ps.label:
            ps.ax.scatter(x, ps.y, s=40, alpha=0.5, c='k', marker='.',
                          label=ps.label)
        else:
            ps.ax.scatter(x, ps.y, s=40, alpha=0.5, c='k', marker='.',
                       label='observation')
    x_extremes = np.empty((4,1,))
    x_extremes[:] = np.nan
    y_extremes = np.empty((4,1,))
    y_extremes[:] = np.nan
    if ps.do_plot_extreme[0]:
        index_min_x = np.argmin(x)
        x_extremes[0] = x[index_min_x]
        y_extremes[0] = y[index_min_x]
    if ps.do_plot_extreme[1]:
        index_max_x = np.argmax(x)
        x_extremes[1] = x[index_max_x]
        y_extremes[1] = y[index_max_x]
    if ps.do_plot_extreme[2]:
        index_min_y = np.argmin(y)
        x_extremes[2] = x[index_min_y]
        y_extremes[2] = y[index_min_y]
    if ps.do_plot_extreme[3]:
        index_max_y = np.argmax(y)
        x_extremes[3] = x[index_max_y]
        y_extremes[3] = y[index_max_y]
    if any(d == True for d in ps.do_plot_extreme):
        ps.ax.scatter(x_extremes, y_extremes,  s=40, c='g', marker='*',
                   label='observed extreme')


def plot_dependence_functions(
        fit, fig, unconditonal_variable_label=None,
        style_discrete_values='r.', style_dependence_function='b-'):
    """

    Parameters
    ----------
    fit : Fit
    fig : Figure
    unconditonal_variable_label : str
    style_discrete_values : str
        Style of the discrete values for the parameters that were derived from
        individual distribution fitting.
    style_dependence_function : str
        Style of the fitted dependence function.
    """
    ax = fig.add_subplot(211)
    scale_at = fit.multiple_fit_inspection_data[1].scale_at
    x1 = np.linspace(0, max(scale_at)*1.1, 100)
    dp_function = '$' + str('%.2f' % fit.mul_var_dist.distributions[1].scale.a) + \
                  '+' + str('%.2f' % fit.mul_var_dist.distributions[1].scale.b) + \
                  '\cdot h_s^{' + str('%.2f' % fit.mul_var_dist.distributions[1].scale.c) + '}$'


    if fit.mul_var_dist.distributions[1].name == 'Lognormal':
        plt.plot(scale_at, np.log(fit.multiple_fit_inspection_data[1].scale_value),
                 style_discrete_values, label='from individual distribution fitting')
        plt.plot(x1, np.log(fit.mul_var_dist.distributions[1].scale(x1)),
                 style_dependence_function, label=dp_function)
        ylabel = '$μ$'
    if fit.mul_var_dist.distributions[1].name == 'Weibull':
        plt.plot(scale_at, fit.multiple_fit_inspection_data[1].scale_value,
                 style_discrete_values, label='from individual distribution fitting')
        plt.plot(x1, fit.mul_var_dist.distributions[1].scale(x1),
                 style_dependence_function, label=dp_function)
        ylabel = '$α_v$'
    #plt.xlabel(unconditonal_variable_label)
    plt.legend(frameon=False)
    plt.ylabel(ylabel)

    ax = fig.add_subplot(212)
    shape_at = fit.multiple_fit_inspection_data[1].shape_at
    x1 = np.linspace(0, max(shape_at)*1.1, 100)
    plt.plot(shape_at, fit.multiple_fit_inspection_data[1].shape_value,
             style_discrete_values)
    plt.plot(x1, fit.mul_var_dist.distributions[1].shape(x1),
             style_dependence_function)
    plt.xlabel(unconditonal_variable_label)
    if fit.mul_var_dist.distributions[1].name == 'Lognormal':
        ylabel = '$σ$'
        dp_function = '$' + str('%.2f' % fit.mul_var_dist.distributions[1].shape.a) + \
                      '+' + str('%.2f' % fit.mul_var_dist.distributions[1].shape.b) + \
                      '\exp (' + str('%.2f' % fit.mul_var_dist.distributions[1].shape.c) + \
                      'h_s)$'
    if fit.mul_var_dist.distributions[1].name == 'Weibull':
        ylabel = '$β_v$'
        dp_function = '$' + str('%.2f' % fit.mul_var_dist.distributions[1].shape.a) + \
                      '+' + str('%.2f' % fit.mul_var_dist.distributions[1].shape.b) + \
                      '\cdot h_s^{' + str('%.2f' % fit.mul_var_dist.distributions[1].shape.c) + '}$'
    plt.legend(['from individual distribution fitting', dp_function], frameon=False)
    plt.ylabel(ylabel)


def plot_contour(x, y, ax, contour_label=None, x_label=None, y_label=None,
                 line_style='b-', alpha=1, plotted_sample=None):
    """
    Plots the environmental contour.

    The method expects the coordinates to be ordered by angle.

    Parameters
    ----------
    x : ndarray of doubles
        The contour's coordinates in the x-direction.
    y : ndarray of doubles
        The contour's coordiantes in the y-direction.
    ax : Axes
        Axes of the figure where the contour should be plotted.
    contour_label : str, optional
        The environmental contour's label that will be used in the legend.
    x_label : str, optional
        Label for the x-axis.
    y_label : str, optional
        Label for the y-axis.
    line_style : str, optional
        Matplotlib line style.
    alpha : float, optional
        Alpha value (transparency) for the contour's line.
    plotted_sample : PlottedSample,
        The sample that should be plotted and its meta information.
    """
    # For generating a closed contour: add the first coordinate at the end.
    xplot = x.tolist()
    xplot.append(x[0])
    yplot = y.tolist()
    yplot.append(y[0])

    # Plot the contour and, if provided, also the sample.
    if plotted_sample:
        plot_sample(plotted_sample)
    if contour_label:
        ax.plot(xplot, yplot, line_style, alpha=alpha, label=contour_label)
    else:
        ax.plot(xplot, yplot, line_style, alpha=alpha)

    # Format the figure.
    if contour_label:
        plt.legend(loc='upper left', frameon=False)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    y_lim_factor = 1.2
    if plotted_sample:
        # If there is not enough space for the legend in the upper left corner:
        # make space for it.
        max_index = np.where(plotted_sample.y == max(plotted_sample.y))
        if plotted_sample.x[max_index] < 0.6 * max(max(x), max(plotted_sample.x)):
            y_lim_factor = 1.35

        upper_ylim = max(max(y), max(plotted_sample.y)) * y_lim_factor
    else:
        upper_ylim = max(y) * y_lim_factor
    plt.ylim((0, upper_ylim))

    # Remove axis on the right and on the top (Matlab 'box off').
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

class PlottedSample():
    """
    Class that holds a plotted sample and its meta information.

    Attributes
    ----------
    x : ndarray of floats
        The sample's first environmental variable.
    y : ndarray of floats
        The sample's second environmental variable.
    ax : Axes
        Axes of the figure where the scatter plot should be drawn.
    label : str
        Label that will be used in the legend for the sample.
    x_inside : ndarray of floats
        Values in the first dimension of the points inside the contour.
    y_inside : ndarray of floats
        Values in the second dimension of the points inside the contour.
    x_outside : ndarray of floats
        Values in the first dimension of the points outside the contour.
    y_outside : ndarray of floats
        Values in the second dimension of the points outside the contour.
    return_period : int, optional
        Return period in years. Is used in legend for describing the inside and
        outside datapoints.
    do_plot_extreme : ndarray of booleans, optional
        Specifies which extremes should be plotted.
        The order is [min(x), max(x), min(y), max(y)].
    """

    def __init__(self, x, y, ax, label=None, x_inside=None, y_inside=None,
                 x_outside=None, y_outside=None, return_period=None,
                 do_plot_extreme=[False, False, False, False]):
        """

        Parameters
        ----------
        x : ndarray of floats
            The sample's first environmental variable.
        y : ndarray of floats
            The sample's second environmental variable.
        ax : Axes
            Axes of the figure where the scatter plot should be drawn.
        label : str
            Label that will be used in the legend for the sample.
        x_inside : ndarray of floats
            Values in the first dimension of the points inside the contour.
        y_inside : ndarray of floats
            Values in the second dimension of the points inside the contour.
        x_outside : ndarray of floats
            Values in the first dimension of the points outside the contour.
        y_outside : ndarray of floats
            Values in the second dimension of the points outside the contour.
        return_period : int, optional
            Return period in years. Is used in legend for describing the inside and
            outside datapoints.
        do_plot_extreme : ndarray of booleans, optional
            Specifies which extremes should be plotted.
            The order is [min(x), max(x), min(y), max(y)].
        """
        self.x = x
        self.y = y
        self.ax = ax
        self.label = label
        self.x_inside = x_inside
        self.y_inside = y_inside
        self.x_outside = x_outside
        self.y_outside = y_outside
        self.return_period = return_period
        self.do_plot_extreme = do_plot_extreme


def plot_confidence_interval(x_median, y_median, x_bottom, y_bottom, x_upper,
                             y_upper, ax, x_label=None, y_label=None,
                             contour_labels=[None, None, None], plotted_sample=None):
    """
    Plots the confidence interval (median, bottom, upper) in a standardized
    appearance.

    Parameters
    ----------
    x_median : ndarray of doubles
        The 50-percentile contour's coordinates in the x-direction.
    y_median : ndarray of doubles
        The 50-percentile contour's coordinates in the y-direction.
    x_bottom : ndarray of doubles
        The bottom percentile contour's coordinates in the x-direction.
    y_bottom : ndarray of doubles
        The bottom percentile contour's coordinates in the y-direction.
    x_upper : ndarray of doubles
        The upper percentile contour's coordinates in the x-direction.
    y_upper : ndarray of doubles
        The upper percentile contour's coordinates in the y-direction.
    ax : Axes
        Axes of the figure where the contour should be plotted.
    x_label : str, optional
        Label for the x-axis.
    y_label : str, optional
        Label for the y-axis.
    contour_labels : list of str, optional
        Label for th environmental contours that will be used in the legend.
    plotted_sample : PlottedSample, optional
        If provided, this sample is drawn together with the contours.
    """
    plot_contour(x=x_median,
                 y=y_median,
                 ax=ax,
                 x_label=x_label,
                 y_label=y_label,
                 line_style='b-',
                 contour_label=contour_labels[0],
                 plotted_sample=plotted_sample)
    plot_contour(x=x_bottom,
                 y=y_bottom,
                 contour_label=contour_labels[1],
                 line_style='r--',
                 ax=ax)
    plot_contour(x=x_upper,
                 y=y_upper,
                 contour_label=contour_labels[2],
                 line_style='r--',
                 ax=ax)
    plt.xlim((0, 32))
    plt.ylim((0, 12))
