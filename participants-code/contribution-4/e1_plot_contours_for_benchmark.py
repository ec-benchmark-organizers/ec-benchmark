import numpy as np
import matplotlib.pyplot as plt


from plot import plot_contour, PlottedSample
from contour_statistics import points_outside
from read_write import read_dataset, read_contour, determine_file_name_e1, \
    determine_file_name_e2

DATASET_CHARS = ['A', 'B', 'C', 'D', 'E', 'F']

for dataset_char in DATASET_CHARS:
    # Read dataset A, B  or C.
    file_path = 'datasets/' + dataset_char + '.txt'
    sample_x, sample_y, label_x, label_y = read_dataset(file_path)

    # Differentiate between sea state and wind wave contours.
    if dataset_char in ('A', 'B', 'C'):
        return_period_long_tr = 20
    else:
        return_period_long_tr = 50

    # Read the contours from the csv files.
    folder_name = 'contour-coordinates/'
    file_name_1 = determine_file_name_e1('Andreas', 'Haselsteiner', dataset_char, 1)
    file_name_long_tr = determine_file_name_e1('Andreas', 'Haselsteiner',
                                               dataset_char,
                                               return_period_long_tr)


    (contour_x_1, contour_y_1) = read_contour(folder_name + file_name_1)
    (contour_x_long, contour_y_long) = read_contour(folder_name + file_name_long_tr)

    # Switch the order of variables for plotting Hs over Tz.
    if dataset_char in ('A', 'B', 'C'):
        sample_x, sample_y = sample_y, sample_x
        label_x, label_y = label_y, label_x
        contour_x_1, contour_y_1 = contour_y_1, contour_x_1
        contour_x_long, contour_y_long = contour_y_long, contour_x_long

    # Find datapoints that exceed the 20/50-yr contour.
    x_outside, y_outside, x_inside, y_inside = \
        points_outside(contour_x_long,
                       contour_y_long,
                       np.asarray(sample_x),
                       np.asarray(sample_y))
    print('Number of points outside the contour: ' +  str(len(x_outside)))

    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(111)

    # Plot the 1-year contour.
    plot_contour(x=contour_x_1,
                 y=contour_y_1,
                 ax=ax,
                 contour_label='1-yr contour',
                 line_style='b--')

    # Plot the 20/50-year contour and the sample.
    plotted_sample = PlottedSample(x=np.asarray(sample_x),
                                   y=np.asarray(sample_y),
                                   ax=ax,
                                   x_inside=x_inside,
                                   y_inside=y_inside,
                                   x_outside=x_outside,
                                   y_outside=y_outside,
                                   return_period=return_period_long_tr)
    plot_contour(x=contour_x_long,
                 y=contour_y_long,
                 ax=ax,
                 contour_label=str(return_period_long_tr) + '-yr contour',
                 x_label=label_x,
                 y_label=label_y,
                 line_style='b-',
                 plotted_sample=plotted_sample,)
    plt.title('Dataset ' + dataset_char)

plt.show()

file_name_long_tr = determine_file_name_e1('Andreas', 'Haselsteiner',
                                           dataset_char,
                                           return_period_long_tr)