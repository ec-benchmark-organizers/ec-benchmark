import numpy as np
import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from plot import plot_contour, PlottedSample
from contour_statistics import points_outside
from read_write import read_dataset, determine_file_name_e1, write_contour, read_contour

# Read dataset D, E or F.
DATASET_CHAR = 'D'
file_path = '../datasets/' + DATASET_CHAR + '.txt'
sample_v, sample_hs, label_v, label_hs= read_dataset(file_path)

# Define the structure of the probabilistic model that will be fitted to the
# dataset. We will use the model that is recommended in DNV-RP-C205 (2010) on
# page 38 and that is called 'conditonal modeling approach' (CMA).
dist_description_hs = {'name': 'Weibull_3p',
                      'dependency': (None, None, None),
                      'width_of_intervals': 0.5}
dist_description_v = {'name': 'Weibull_2p',
                      'dependency': (0,  None, 0), #Shape, Location, Scale
                      'functions': ('power3', None, 'power3') #Shape, Location, Scale
                      }

# Fit the model to the data.
my_fit = Fit((sample_hs, sample_v), (dist_description_hs, dist_description_v))
dist0 = my_fit.mul_var_dist.distributions[0]
print('First variable: ' + dist0.name + ' with '
      + ' scale: ' + str(dist0.scale) + ', '
      + ' shape: ' + str(dist0.shape) + ', '
      + ' location: ' + str(dist0.loc))
dist1 = my_fit.mul_var_dist.distributions[1]
print('Second variable: ' + dist1.name + ' with '
      + ' scale: ' + str(dist1.scale) + ', '
      + ' shape: ' + str(dist1.shape) + ', '
      + ' location: ' + str(dist1.loc))

# Compute IFORM-contours with return periods of 1 and 50 years.
return_period_1 = 1
iform_contour_1 = IFormContour(my_fit.mul_var_dist, return_period_1, 1, 100)
return_period_50 = 50
iform_contour_50 = IFormContour(my_fit.mul_var_dist, return_period_50, 1, 100)

# Save the contours as csv files in the required format.
folder_name = 'contour_coordinates/'
file_name_1 = determine_file_name_e1('John', 'Doe', DATASET_CHAR, return_period_1)
write_contour(iform_contour_1.coordinates[0][0],
              iform_contour_1.coordinates[0][1],
              folder_name + file_name_1,
              label_x=label_hs,
              label_y=label_v)
file_name_50 = determine_file_name_e1('John', 'Doe', DATASET_CHAR, return_period_50)
write_contour(iform_contour_50.coordinates[0][0],
              iform_contour_50.coordinates[0][1],
              folder_name + file_name_50,
              label_x=label_hs,
              label_y=label_v)

# Read the contour coordinates from the created csv files.
(contour_hs_1, contour_v_1) = read_contour(folder_name + file_name_1)
(contour_hs_50, contour_v_50) = read_contour(folder_name + file_name_50)

# Find datapoints that exceed the contour.
hs_outside, v_outside, hs_inside, v_inside = \
    points_outside(contour_hs_50,
                   contour_v_50,
                   np.asarray(sample_hs),
                   np.asarray(sample_v))
print('Number of points outside the contour: ' +  str(len(hs_outside)))

fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)

# Plot the 1-year contour.
plot_contour(x=contour_v_1,
             y=contour_hs_1,
             ax=ax,
             return_period=return_period_1,
             x_label=label_v,
             y_label=label_hs,
             line_style='b--')

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
             return_period=return_period_50,
             x_label=label_v,
             y_label=label_hs,
             line_style='b-',
             plotted_sample=plotted_sample)
plt.title('Dataset ' + DATASET_CHAR)

plt.show()
