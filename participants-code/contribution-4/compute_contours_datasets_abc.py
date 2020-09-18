import numpy as np
import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import HighestDensityContour as HDC
from plot import plot_contour, PlottedSample, plot_marginal_fit, plot_dependence_functions
from contour_statistics import points_outside, sort_points_to_form_continous_line
from read_write import read_dataset, determine_file_name_e1, write_contour, read_contour

# Read dataset A, B  or C.
DATASET_CHAR = 'A'
file_path = 'datasets/' + DATASET_CHAR + '.txt'
sample_hs, sample_tz, label_hs, label_tz= read_dataset(file_path)

# Define the structure of the probabilistic model that will be fitted to the
# dataset.
dist_description_hs = {'name': 'Weibull_Exp',
                      'dependency': (None, None, None, None),
                      'width_of_intervals': 0.5}
dist_description_tz = {'name': 'Lognormal_SigmaMu',
                      'dependency': (0,  None, 0), #Shape, Location, Scale
                      'functions': ('asymdecrease3', None, 'lnsquare2'), #Shape, Location, Scale
                      'min_datapoints_for_fit': 50
                      }

# Fit the model to the data.
fit = Fit((sample_hs, sample_tz), (
    dist_description_hs, dist_description_tz))
dist0 = fit.mul_var_dist.distributions[0]

fig = plt.figure(figsize=(12.5, 3.5), dpi=150)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
plot_marginal_fit(sample_hs, dist0, fig=fig, ax=ax1, label='$h_s$ (m)',
                  dataset_char=DATASET_CHAR)
plot_dependence_functions(fit=fit, fig=fig, ax1=ax2, ax2=ax3, unconditonal_variable_label=label_hs)
fig.suptitle('Dataset ' + DATASET_CHAR)
fig.subplots_adjust(wspace=0.25, bottom=0.15)

# Compute highest density contours with return periods of 1 and 20 years.
return_period_1 = 1
ts = 1 # Sea state duration in hours.
limits = [(0, 20), (0, 20)] # Limits of the computational domain.
deltas = [0.05, 0.05] # Dimensions of the grid cells.
hdc_contour_1 = HDC(fit.mul_var_dist, return_period_1, ts, limits, deltas)
return_period_20 = 20
hdc_contour_20 = HDC(fit.mul_var_dist, return_period_20, ts, limits, deltas)

c = sort_points_to_form_continous_line(hdc_contour_1.coordinates[0][0],
                                       hdc_contour_1.coordinates[0][1],
                                       do_search_for_optimal_start=True)
contour_hs_1 = c[0]
contour_tz_1 = c[1]

c = sort_points_to_form_continous_line(hdc_contour_20.coordinates[0][0],
                                       hdc_contour_20.coordinates[0][1],
                                       do_search_for_optimal_start=True)
contour_hs_20 = c[0]
contour_tz_20 = c[1]

# Save the contours as csv files in the required format.
folder_name = 'contour-coordinates/'
file_name_1 = determine_file_name_e1('Andreas', 'Haselsteiner', DATASET_CHAR, return_period_1)
write_contour(contour_hs_1,
              contour_tz_1,
              folder_name + file_name_1,
              label_x=label_hs,
              label_y=label_tz)
file_name_20 = determine_file_name_e1('Andreas', 'Haselsteiner', DATASET_CHAR, return_period_20)
write_contour(contour_hs_20,
              contour_tz_20,
              folder_name + file_name_20,
              label_x=label_hs,
              label_y=label_tz)

# Read the contours from the csv files.
(contour_hs_1, contour_tz_1) = read_contour(folder_name + file_name_1)
(contour_hs_20, contour_tz_20) = read_contour(folder_name + file_name_20)

# Find datapoints that exceed the 20-yr contour.
hs_outside, tz_outside, hs_inside, tz_inside = \
    points_outside(contour_hs_20,
                   contour_tz_20,
                   np.asarray(sample_hs),
                   np.asarray(sample_tz))
print('Number of points outside the contour: ' +  str(len(hs_outside)))

fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)

# Plot the 1-year contour.
plot_contour(x=contour_tz_1,
             y=contour_hs_1,
             ax=ax,
             contour_label=str(return_period_1) + '-yr contour',
             line_style='b--')

# Compute the median tz conditonal on hs.
hs = np.linspace(0, 14, 100)
d1 = fit.mul_var_dist.distributions[1]
c1 = d1.scale.a
c2 = d1.scale.b
tz = c1 + c2 * np.sqrt(np.divide(hs, 9.81))

# Plot the 20-year contour and the sample.
plotted_sample = PlottedSample(x=np.asarray(sample_tz),
                               y=np.asarray(sample_hs),
                               ax=ax,
                               x_inside=tz_inside,
                               y_inside=hs_inside,
                               x_outside=tz_outside,
                               y_outside=hs_outside,
                               return_period=return_period_20)
plot_contour(x=contour_tz_20,
             y=contour_hs_20,
             ax=ax,
             contour_label=str(return_period_20) + '-yr contour',
             x_label=label_tz,
             y_label=label_hs,
             line_style='b-',
             plotted_sample=plotted_sample,
             x_lim=(0, 19),
             upper_ylim=15,
             median_x=tz,
             median_y=hs,
             median_label='median of $T_z | H_s$')
plt.title('Dataset ' + DATASET_CHAR)
plt.show()
