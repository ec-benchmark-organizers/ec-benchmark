import numpy as np
import matplotlib.pyplot as plt

from plot import PlottedSample, plot_confidence_interval
from read_write import read_dataset, determine_file_name_e2, read_contour

NR_OF_YEARS_TO_DRAW = 1 # Must be 1, 5 or 25.
FIRST_NAME = 'Andreas'
LAST_NAME = 'Haselsteiner'

# Read dataset D.
file_path = 'datasets/D.txt'
dataset_d_v, dataset_d_hs, label_v, label_hs = read_dataset(file_path)

# Read the contours that have beem computed previously from csv files.
folder_name = 'contour-coordinates/'
file_name_median = determine_file_name_e2(
    FIRST_NAME, LAST_NAME, NR_OF_YEARS_TO_DRAW, 'median')
file_name_bottom = determine_file_name_e2(
    FIRST_NAME, LAST_NAME, NR_OF_YEARS_TO_DRAW, 'bottom')
file_name_upper = determine_file_name_e2(
    FIRST_NAME, LAST_NAME, NR_OF_YEARS_TO_DRAW, 'upper')
(contour_v_median, contour_hs_median) = read_contour(
    folder_name + file_name_median)
(contour_v_bottom, contour_hs_bottom) = read_contour(
    folder_name + file_name_bottom)
(contour_v_upper, contour_hs_upper) = read_contour(
    folder_name + file_name_upper)

# Plot the sample, the median contour and the confidence interval.
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)
plotted_sample = PlottedSample(x=np.asarray(dataset_d_v),
                               y=np.asarray(dataset_d_hs),
                               ax=ax,
                               label='dataset D')
contour_labels = ['50th percentile contour', '2.5th percentile contour',
                  '97.5th percentile contour']
plot_confidence_interval(
    x_median=contour_v_median, y_median=contour_hs_median,
    x_bottom=contour_v_bottom, y_bottom=contour_hs_bottom,
    x_upper=contour_v_upper, y_upper=contour_hs_upper, ax=ax, x_label=label_v,
    y_label=label_hs, contour_labels=contour_labels, plotted_sample=plotted_sample)
if NR_OF_YEARS_TO_DRAW == 1:
    plt.title('Samples cover ' + str(NR_OF_YEARS_TO_DRAW) + ' year')
else:
    plt.title('Samples cover ' + str(NR_OF_YEARS_TO_DRAW) + ' years')
plt.show()