import numpy as np
import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import HighestDensityContour as HDC
from plot import plot_contour, PlottedSample, plot_marginal_fit, plot_dependence_functions
from contour_statistics import points_outside, sort_points_to_form_continous_line
from read_write import read_dataset, determine_file_name_e1, write_contour, read_contour

# Read dataset D, E  or F.
DATASET_CHAR = 'D'
file_path = 'datasets/' + DATASET_CHAR + '.txt'
sample_v, sample_hs, label_v, label_hs = read_dataset(file_path)
label_v = 'wind speed (m s$^{-1}$)'

# Define the structure of the probabilistic model that will be fitted to the
# dataset.
dist_description_v = {'name': 'Weibull_Exp',
                      'dependency': (None, None, None, None),
                      'width_of_intervals': 2}
dist_description_hs = {'name': 'Weibull_Exp',
                       'fixed_parameters': (None, None, None, 5),
                       # shape, location, scale, shape2
                       'dependency': (0, None, 0, None),
                       # shape, location, scale, shape2
                       'functions': ('logistics4', None, 'alpha3', None),
                       # shape, location, scale, shape2
                       'min_datapoints_for_fit': 50,
                       'do_use_weights_for_dependence_function': True}


# Fit the model to the data.
fit = Fit((sample_v, sample_hs),
          (dist_description_v, dist_description_hs))

dist0 = fit.mul_var_dist.distributions[0]

fig = plt.figure(figsize=(12.5, 3.5), dpi=150)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
plot_marginal_fit(sample_v, dist0, fig=fig, ax=ax1, label='$v$ (m s$^{-1}$)',
                  dataset_char=DATASET_CHAR)
plot_dependence_functions(fit=fit, fig=fig, ax1=ax2, ax2=ax3, unconditonal_variable_label=label_v)
fig.suptitle('Dataset ' + DATASET_CHAR)
fig.subplots_adjust(wspace=0.25, bottom=0.15)

# Compute highest density contours with return periods of 1 and 50 years.
return_period_1 = 1
ts = 1 # Sea state duration in hours.
limits = [(0, 45), (0, 20)] # Limits of the computational domain.
deltas = [0.05, 0.05] # Dimensions of the grid cells.
hdc_contour_1 = HDC(fit.mul_var_dist, return_period_1, ts, limits, deltas)
return_period_50 = 50
hdc_contour_50 = HDC(fit.mul_var_dist, return_period_50, ts, limits, deltas)

c = sort_points_to_form_continous_line(hdc_contour_1.coordinates[0][0],
                                       hdc_contour_1.coordinates[0][1],
                                       do_search_for_optimal_start=True)
contour_v_1 = c[0]
contour_hs_1 = c[1]

c = sort_points_to_form_continous_line(hdc_contour_50.coordinates[0][0],
                                       hdc_contour_50.coordinates[0][1],
                                       do_search_for_optimal_start=True)
contour_v_50 = c[0]
contour_hs_50 = c[1]

# Save the contours as csv files in the required format.
folder_name = 'contour-coordinates/'
file_name_1 = determine_file_name_e1('Andreas', 'Haselsteiner', DATASET_CHAR, return_period_1)
write_contour(contour_v_1,
              contour_hs_1,
              folder_name + file_name_1,
              label_x=label_v,
              label_y=label_hs)
file_name_20 = determine_file_name_e1('Andreas', 'Haselsteiner', DATASET_CHAR, return_period_50)
write_contour(contour_v_50,
              contour_hs_50,
              folder_name + file_name_20,
              label_x=label_v,
              label_y=label_hs)

# Read the contours from the csv files.
(contour_v_1, contour_hs_1) = read_contour(folder_name + file_name_1)
(contour_v_50, contour_hs_50) = read_contour(folder_name + file_name_20)

# Find datapoints that exceed the 20-yr contour.
v_outside, hs_outside, v_inside, hs_inside = \
    points_outside(contour_v_50,
                   contour_hs_50,
                   np.asarray(sample_v),
                   np.asarray(sample_hs))
print('Number of points outside the contour: ' +  str(len(v_outside)))

fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)

# Plot the 1-year contour.
plot_contour(x=contour_v_1,
             y=contour_hs_1,
             ax=ax,
             contour_label=str(return_period_1) + '-yr contour',
             line_style='b--')

# Compute the median hs conditonal on v.
x = np.linspace(0, 35, 100)
d1 = fit.mul_var_dist.distributions[1]
a = d1.scale.a
b = d1.scale.b
c = d1.scale.c
y = a + b * np.power(x, c)

# Plot the 50-year contour and the sample.
plotted_sample = PlottedSample(x=np.asarray(sample_v),
                               y=np.asarray(sample_hs),
                               ax=ax,
                               x_inside=v_inside,
                               y_inside=hs_inside,
                               x_outside=v_outside,
                               y_outside=hs_outside,
                               return_period=return_period_50)
plot_contour(x=contour_v_50,
             y=contour_hs_50,
             ax=ax,
             contour_label=str(return_period_50) + '-yr contour',
             x_label=label_v,
             y_label=label_hs,
             line_style='b-',
             plotted_sample=plotted_sample,
             x_lim=(0, 35),
             upper_ylim=20,
             median_x=x,
             median_y=y,
             median_label='median of $H_s | V$')

plt.title('Dataset ' + DATASET_CHAR)
plt.show()
