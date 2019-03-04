import numpy as np
import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from plot import plot_contour, PlottedSample
from read_write import read_dataset, determine_file_name_e2, write_contour, read_contour
from contour_intersection import contour_intersection

# Define the number of years of data that one bootstrap sample should contain.
# In the exercise 1, 5 and 25 years are used.
NR_OF_YEARS_TO_DRAW = 1

NR_OF_BOOTSTRAP_SAMPLES = 1000
BOTTOM_PERCENTILE = 2.5
UPPER_PERCENTILE = 97.5
DO_COMPUTE_CONFIDENCE_INTERVAL = True
PLOT_ANGLE_LINES = False
NR_OF_POINTS_ON_CONTOUR = 100

# Read dataset D.
file_path = '../datasets/D.txt'
dataset_d_v, dataset_d_hs, label_v, label_hs = read_dataset(file_path)

# Define the origin (will be used to compute confidence intervals).
v0 = np.median(dataset_d_v)
hs0 = np.median(dataset_d_hs)

nr_of_datapoints_to_draw = int(NR_OF_YEARS_TO_DRAW * 365.25 * 24)
np.random.seed(9001)

for i in range(NR_OF_BOOTSTRAP_SAMPLES):
    # Resample from the hindcast dataset to get the sample D_i.
    sample_indices = np.random.randint(dataset_d_v.size, size=nr_of_datapoints_to_draw)
    v_i = np.take(dataset_d_v, sample_indices)
    hs_i = np.take(dataset_d_hs, sample_indices)

    # Define the structure of the probabilistic model that will be fitted to the
    # dataset. We will use the model that is recommended in DNV-RP-C205 (2010) on
    # page 38 and that is called 'conditonal modeling approach' (CMA).
    dist_description_hs = {'name': 'Weibull_3p',
                           'dependency': (None, None, None),
                           'width_of_intervals': 0.5}
    dist_description_v = {'name': 'Weibull_2p',
                          'dependency': (0, None, 0),  # Shape, Location, Scale
                          'functions': ('power3', None, 'power3')
                          # Shape, Location, Scale
                          }

    # Fit the model to the dataset.
    my_fit = Fit((hs_i, v_i), (dist_description_hs, dist_description_v))
    dist0 = my_fit.mul_var_dist.distributions[0]
    dist1 = my_fit.mul_var_dist.distributions[1]

    # Compute 50-yr IFORM contour.
    return_period = 50
    iform_contour_i = IFormContour(my_fit.mul_var_dist, return_period, 1,
                                   NR_OF_POINTS_ON_CONTOUR)

    if DO_COMPUTE_CONFIDENCE_INTERVAL:
        # Define angles based on normalization. Give the first quadrant of the
        # normalized angle. Based on it the full 360° will be calculated.
        theta_stars = np.linspace(0.0, 90.0, 37) / 180 * np.pi
        t1 = max(dataset_d_v) - min(dataset_d_v)
        t2 = max(dataset_d_hs) - min(dataset_d_hs)
        q1 = np.arctan(np.tan(theta_stars) * t2 / t1)
        q2 = np.pi - q1
        q2 = np.sort(q2)
        q2 = q2[1:]
        q3 = np.pi + q1
        q3 = q3[1:]
        q4 = 2 * np.pi - q1
        q4 = np.sort(q4)
        q4 = q4[1:-2]
        thetas = np.concatenate([q1, q2, q3, q4], axis=0)
        #print('Thetas: ' + str(thetas / np.pi * 180))
        #print('Theta_stars: ' + str(theta_stars/np.pi * 180))
        nr_of_datapoints_on_angled_line = 10
        line_tot_length = 50.0
        line_length = np.linspace(0.0, line_tot_length, nr_of_datapoints_on_angled_line)

        # Compute lines that have an angle theta to the x-axis.
        theta_line_v = list()
        theta_line_hs = list()
        iform_contour_i.theta_v = list()
        iform_contour_i.theta_hs = list()
        for j, theta in enumerate(thetas):
            theta_line_v.append(np.multiply(np.cos(theta),  line_length) + v0)
            theta_line_hs.append(np.multiply(np.sin(theta), line_length) + hs0)
            c_v = np.append(iform_contour_i.coordinates[0][1], iform_contour_i.coordinates[0][1][0])
            c_hs = np.append(iform_contour_i.coordinates[0][0], iform_contour_i.coordinates[0][0][0])
            theta_v_j, theta_hs_j = contour_intersection(
                theta_line_v[j], theta_line_hs[j], c_v, c_hs, True)

            iform_contour_i.theta_v.append(theta_v_j)
            iform_contour_i.theta_hs.append(theta_hs_j)
    if i == 0:
        contours = [iform_contour_i]
    else:
        contours.append(iform_contour_i)

# Plot the environmental contours.
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)
for i, contour in enumerate(contours):
    if i == 0:
        plotted_sample = PlottedSample(x=np.asarray(dataset_d_v),
                                       y=np.asarray(dataset_d_hs),
                                       ax=ax,
                                       label='dataset D')
        plot_contour(x=contour.coordinates[0][1],
                     y=contour.coordinates[0][0],
                     ax=ax,
                     return_period=return_period,
                     x_label=label_v,
                     y_label=label_hs,
                     line_style='b-',
                     alpha=0.4,
                     plotted_sample=plotted_sample)
    else:
        plot_contour(x=contour.coordinates[0][1],
                     y=contour.coordinates[0][0],
                     line_style='b-',
                     alpha=0.4,
                     ax=ax)
    if DO_COMPUTE_CONFIDENCE_INTERVAL and PLOT_ANGLE_LINES:
        for j, (line_v, line_hs) in enumerate(zip(theta_line_v, theta_line_hs)):
            if i == 0:
                plt.plot(line_v, line_hs, 'r-')
            plt.plot(contour.theta_v, contour.theta_hs, 'gx')
if NR_OF_YEARS_TO_DRAW == 1:
    plt.title('Samples cover ' + str(NR_OF_YEARS_TO_DRAW) + ' year')
else:
    plt.title('Samples cover ' + str(NR_OF_YEARS_TO_DRAW) + ' years')
plt.xlim((0, 32))
plt.ylim((0, 12))
plt.show()

if DO_COMPUTE_CONFIDENCE_INTERVAL:
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
    percentile50_index = int(round((NR_OF_BOOTSTRAP_SAMPLES - 1) * (50.0 / 100.0)))
    bottom_percentile_index = int(round((NR_OF_BOOTSTRAP_SAMPLES - 1) * (BOTTOM_PERCENTILE / 100.0)))
    upper_percentile_index = int(round((NR_OF_BOOTSTRAP_SAMPLES - 1) * (UPPER_PERCENTILE / 100.0)))

    # Save the median, bottom and upper percentile contours.
    folder_name = 'contour_coordinates/'
    file_name_median = determine_file_name_e2(
        'John', 'Doe', NR_OF_YEARS_TO_DRAW, 'median')
    write_contour(sorted_v[percentile50_index, :],
                  sorted_hs[percentile50_index, :],
                  folder_name + file_name_median,
                  label_x=label_v,
                  label_y=label_hs)
    folder_name = 'contour_coordinates/'
    file_name_bottom = determine_file_name_e2(
        'John', 'Doe', NR_OF_YEARS_TO_DRAW, 'bottom')
    write_contour(sorted_v[bottom_percentile_index, :],
                  sorted_hs[bottom_percentile_index, :],
                  folder_name + file_name_bottom,
                  label_x=label_v,
                  label_y=label_hs)
    folder_name = 'contour_coordinates/'
    file_name_upper = determine_file_name_e2(
        'John', 'Doe', NR_OF_YEARS_TO_DRAW, 'upper')
    write_contour(sorted_v[upper_percentile_index, :],
                  sorted_hs[upper_percentile_index, :],
                  folder_name + file_name_upper,
                  label_x=label_v,
                  label_y=label_hs)

    # Read the contours from the csv files.
    (contour_v_median, contour_hs_median) = read_contour(folder_name + file_name_median)
    (contour_v_bottom, contour_hs_bottom) = read_contour(folder_name + file_name_bottom)
    (contour_v_upper, contour_hs_upper) = read_contour(folder_name + file_name_upper)

    # Plot the sample, the median contour and the confidence interval.
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(111)
    plotted_sample = PlottedSample(x=np.asarray(dataset_d_v),
                                   y=np.asarray(dataset_d_hs),
                                   ax=ax,
                                   label='dataset D')
    plot_contour(x=contour_v_median,
                 y=contour_hs_median,
                 ax=ax,
                 return_period=return_period,
                 x_label=label_v,
                 y_label=label_hs,
                 line_style='k--',
                 plotted_sample=plotted_sample)
    plot_contour(x=contour_v_bottom,
                 y=contour_hs_bottom,
                 line_style='r-',
                 ax=ax)
    plot_contour(x=contour_v_upper,
                 y=contour_hs_upper,
                 line_style='g-',
                 ax=ax)
    plt.xlim((0, 32))
    plt.ylim((0, 12))
    plt.show()
