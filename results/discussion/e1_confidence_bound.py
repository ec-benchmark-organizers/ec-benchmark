import matplotlib.pyplot as plt
import numpy as np
from palettable.colorbrewer.qualitative import Paired_9 as mycorder
from shapely.geometry import LineString, Point, MultiPoint
import warnings
from viroconcom.contours import Contour
from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.plot import plot_contour, plot_confidence_interval

dataset_char = 'D'
return_period =  50

lastname_firstname = [
    'Wei_Bernt',
    'GC_CGS',
    'hannesdottir_asta',
    'haselsteiner_andreas',
    'BV',
    'mackay_ed',
    'qiao_chi',
    'rode_anna',
    'vanem_DirectSampling',
]
n_contours_to_analyze = len(lastname_firstname)

class Object(object):
    pass


# Code from https://github.com/ahaselsteiner/2020-paper-omae-hierarchical-models/blob/master/contour_intersection.py
def contour_intersection(contour_x, contour_y, line_x, line_y, do_plot_in_case_of_failure=False):
    """
    Computes the intersection point between two lines.
    The first line should be the environmental contour and the second line a
    straight line. If there are more multiple intersections the point that has
    the longest distance to the origin is returned (assuming the origin is at
    the first point of the straight line).
    Parameters
    ----------
    contour_x : ndarray of floats,
        Coordinates in the first dimension of the contour.
    contour_y : ndarray of floats
        Coordinates in the second dimension of the contour.
    line_x : ndarray of floats
        Coordinates in the first dimension of the straight line.
    line_y : ndarray of floats
        Coordaintes in the second dimension of the straight line.
    do_plot_in_case_of_failure : Boolean,
        if True a plot of the two lines for which no intersection could be
        found is shown.
    Returns
    -------
    x : float
        Intersection coordinate in the first dimension.
    y : float
        Intersection coordinate in the second dimension.
    """

    point_list_l1 = list()
    for x,y in zip(contour_x, contour_y):
        point = Point(x, y)
        point_list_l1.append(point)
    contour = LineString(point_list_l1)

    point_list_l2 = list()
    for x,y in zip(line_x, line_y):
        point = Point(x, y)
        point_list_l2.append(point)
    line2 = LineString(point_list_l2)

    intersection = contour.intersection(line2)
    if type(intersection) is Point:
        return intersection.x, intersection.y
    if type(intersection) is MultiPoint:
        if len(intersection.geoms) > 0:
            print(str(len(intersection.geoms)) + ' intersections were found.'
                  + ' Using the intersection that is the farthest'
                  + ' away from the origin.')
        origin = Point(line_x[0], line_y[0])
        for i, p in enumerate(intersection.geoms):
            if i == 0:
                inter_x = p.x
                inter_y = p.y
                longest_distance = origin.distance(p)
            else:
                if origin.distance(p) > longest_distance:
                    inter_x = p.x
                    inter_y = p.y
        return inter_x, inter_y
    else:
        print('The result is: ' + str(intersection))
        warnings.warn('No point of intersection could be found. '
                      'Returning (nan, nan).', UserWarning)
        if do_plot_in_case_of_failure:
            fig = plt.figure(figsize=(5, 5), dpi=150)
            fig.add_subplot(111)
            plt.plot(contour_x, contour_y, 'b-')
            plt.plot(line_x, line_y, 'r-')
            plt.show()
        return np.nan, np.nan

# Code from: https://github.com/ahaselsteiner/2020-paper-omae-hierarchical-models/blob/master/contour_statistics.py#L44
def thetastar_to_theta(thetastar, xspread, yspread):
    """
    Parameters
    ----------
    thetastar : ndarray of floats
        Angle in the normalized coordinate system.
    xspread : float
        Spread of x (xmax - ymin).
    yspread : float
        Spread of y (ymax - amin).
    Returns
    -------
    theta : float,
        The angle theta in the original coordinate system. The angle is
        defined counter clockwise, 0 at (x=1, y=0) and is converted to be
        inside the interval [0 360).
    """
    theta = np.arctan2(np.sin(thetastar) * yspread, np.cos(thetastar) * xspread)
    for i, t in enumerate(theta):
        if t < 0:
            theta[i] = t + 2 * np.pi
    return theta

colors_for_contribution = mycorder.mpl_colors
for idx in range(2):
        colors_for_contribution.append(colors_for_contribution[8])
colors_for_contribution.append('blue')


fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
max_hs_of_sample = 0

# Load the environmental data.
file_name_provided = 'datasets/' + dataset_char + '.txt'
v_p, hs_p, label_v, label_hs = read_ecbenchmark_dataset(file_name_provided)
max_hs_of_sample = max([max_hs_of_sample, max(hs_p)])

contours = []
contours_v = []
contours_hs = []
max_hs_on_contours = np.empty(n_contours_to_analyze)
for i in range(n_contours_to_analyze):
    contribution_nr = i + 1
    if 11 >= contribution_nr >= 9:
        contribution_nr = 9
    elif contribution_nr > 11:
        # Because contribution 9 holds 3 sets of contours.
        contribution_nr = contribution_nr - 2
    folder_name = 'results/exercise-1/contribution-' + str(contribution_nr)
    file_name = folder_name + '/' + lastname_firstname[i] + '_dataset_' + \
                dataset_char + '_' + str(return_period) + '.txt'
    if contribution_nr in (1, 2, 3, 5, 6, 8, 10):
        (hs, v) = read_contour(file_name)
    else:
        (v, hs) = read_contour(file_name)
    contour = Object()
    contour.c = (np.append(v, v[0]), np.append(hs, hs[0]))
    contours.append(contour)
    contours_v.append(v)
    contours_hs.append(hs)
    max_hs_on_contours[i] = max(hs[~np.isnan(hs)])

# Compute the min, max and median contour.
# First, define the origin and angles based on normalization.
v0 = np.mean(v_p)
hs0 = np.mean(hs_p)
angle_step_for_ci = 2
theta_stars = np.arange(0, 360, angle_step_for_ci) / 180 * np.pi
t1 = max(v_p) - min(v_p)
t2 = max(hs_p) - min(hs_p)
thetas = thetastar_to_theta(theta_stars, t1, t2)
nr_of_datapoints_on_angled_line = 10
line_tot_length = 50.0
line_length = np.linspace(0.0, line_tot_length, nr_of_datapoints_on_angled_line)

# Then, compute lines that have an angle theta to the x-axis.
for i, contour in enumerate(contours):
    theta_line_v = list()
    theta_line_hs = list()
    contour.theta_v = list()
    contour.theta_hs = list()
    for j, theta in enumerate(thetas):
        theta_line_v.append(np.multiply(np.cos(theta),  line_length) + v0)
        theta_line_hs.append(np.multiply(np.sin(theta), line_length) + hs0)
        theta_v_j, theta_hs_j = contour_intersection(
            theta_line_v[j], theta_line_hs[j], contour.c[0], contour.c[1], True)

        contour.theta_v.append(theta_v_j)
        contour.theta_hs.append(theta_hs_j)


theta_v_ij = np.zeros(shape=(len(contours), thetas.size))
theta_hs_ij = np.zeros(shape=(len(contours), thetas.size))
distance_to_origin_ij = np.zeros(shape=(len(contours), thetas.size))
for i, contour in enumerate(contours):
    for j, (v_j, hs_j) in enumerate(zip(contour.theta_v, contour.theta_hs)):
        theta_v_ij[i, j] = v_j
        theta_hs_ij[i, j] = hs_j
        o = np.array([v0, hs0])
        p = np.array([v_j, hs_j]).flatten()
        op = p - o
        distance_to_origin_ij[i, j] = np.sqrt(op[0]*op[0] + op[1]*op[1])
sorted_v = np.zeros(shape=(len(contours), thetas.size))
sorted_hs = np.zeros(shape=(len(contours), thetas.size))
for j in range(thetas.size):
    sorted_indices = np.argsort(distance_to_origin_ij[:, j])
    sorted_v[:, j] = theta_v_ij[sorted_indices, j]
    sorted_hs[:, j] = theta_hs_ij[sorted_indices, j]
lower_index = 0
median_index = 4
upper_index = 8

cl = Object()
cm = Object()
cu = Object()

cl.v = sorted_v[lower_index, :]
cl.hs = sorted_hs[lower_index, :]
cm.v = sorted_v[median_index, :]
cm.hs = sorted_hs[median_index, :]
cu.v = sorted_v[upper_index, :]
cu.hs = sorted_hs[upper_index, :]


# Create the overlay plot.
axs[0].scatter(v_p, hs_p, c='black', alpha=0.5, zorder=-2, rasterized=True, label='Dataset D (provided)')
for i in range(n_contours_to_analyze):
    if i == 0:
        clabel = 'Submitted 50-yr contours'
    else:
        clabel = None
    plot_contour(contours_v[i], contours_hs[i],
                    alpha=0.5, ax=axs[0], contour_label=clabel)

axs[0].set_rasterization_zorder(-1)
axs[0].set_xlabel(label_v.capitalize())
axs[0].set_ylabel(label_hs.capitalize())

# Create the confidence bound plot.
axs[1].scatter(v_p, hs_p, c='black', alpha=0.5, zorder=-2, rasterized=True)
plot_confidence_interval(x_median=cm.v, y_median=cm.hs, 
    x_bottom=cl.v, y_bottom=cl.hs, x_upper=cu.v, y_upper=cu.hs, ax=axs[1], 
    x_label=label_v.capitalize(), y_label=label_hs.capitalize(), 
    contour_labels=['Median contour', 'Min. contour', 'Max. contour'])
 
for ax in axs.flat:
    ax.label_outer()
    ax.set_xlim((0, 35))
    ax.set_ylim((0, 18))
fig.tight_layout()
fig.savefig('results/discussion/gfx/e1_confidence_bounds.pdf', bbox_inches='tight')
