import numpy as np
import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from plot import plot_contour
from contour_statistics import points_outside
from read_write import read_dataset, determine_file_name, write_contour, read_contour

# Read dataset A, B  or C.
DATASET_CHAR = 'A'
file_path = '../datasets/' + DATASET_CHAR + '.txt'
sample_hs, sample_tz, label_hs, label_tz= read_dataset(file_path)

# Define the structure of the probabilistic model that will be fitted to the
# dataset. We will use the model that is recommended in DNV-RP-C205 (2010) on
# page 38 and that is called 'conditonal modeling approach' (CMA).
dist_description_hs = {'name': 'Weibull_3p',
                      'dependency': (None, None, None),
                      'width_of_intervals': 0.5}
dist_description_tz = {'name': 'Lognormal_SigmaMu',
                      'dependency': (0,  None, 0), #Shape, Location, Scale
                      'functions': ('exp3', None, 'power3') #Shape, Location, Scale
                      }

# Fit the model to the data.
my_fit = Fit((sample_hs, sample_tz), (
    dist_description_hs, dist_description_tz))
dist0 = my_fit.mul_var_dist.distributions[0]
print('First variable: ' + dist0.name + ' with '
      + ' scale: ' + str(dist0.scale) + ', '
      + ' shape: ' + str(dist0.shape) + ', '
      + ' location: ' + str(dist0.loc))
print('Second variable: ' + str(my_fit.mul_var_dist.distributions[1]))

# Compute a IFORM-contours with a return periods of 1 and 20 years.
return_period_1 = 1
iform_contour_1 = IFormContour(my_fit.mul_var_dist, return_period_1, 1, 100)
return_period_20 = 20
iform_contour_20 = IFormContour(my_fit.mul_var_dist, return_period_20, 1, 100)

# Save the contours as a csv files in the required format.
file_name_1 = determine_file_name('John', 'Doe', DATASET_CHAR, return_period_1)
write_contour(iform_contour_1.coordinates[0][0],
              iform_contour_1.coordinates[0][1],
              file_name_1,
              label_x=label_hs,
              label_y=label_tz)
file_name_20 = determine_file_name('John', 'Doe', DATASET_CHAR, return_period_20)
write_contour(iform_contour_20.coordinates[0][0],
              iform_contour_20.coordinates[0][1],
              file_name_20,
              label_x=label_hs,
              label_y=label_tz)
# Read the contours from the csv files.
(contour_hs_1, contour_tz_1) = read_contour(file_name_1)
(contour_hs_20, contour_tz_20) = read_contour(file_name_20)

# Find datapoints that exceed the 20-yr contour.
hs_outside, tz_outside, hs_inside, tz_inside = \
    points_outside(contour_hs_20,
                   contour_tz_20,
                   np.asarray(sample_hs),
                   np.asarray(sample_tz))
print('Number of points outside the contour: ' +  str(len(hs_outside)))

# Plot the contour.
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)
# Plot the 1-year contour.
plot_contour(x=contour_tz_1,
             y=contour_hs_1,
             return_period=return_period_1,
             ax=ax,
             x_label=label_tz,
             y_label=label_hs,
             line_style='b--')
# Plot the 20-year contour and the sample.
sample_struct = [np.asarray(sample_tz),
                 np.asarray(sample_hs),
                 tz_inside,
                 hs_inside,
                 tz_outside,
                 hs_outside,
                 return_period_20,
                 [False, False, False, False]]
plot_contour(x=contour_tz_20,
             y=contour_hs_20,
             return_period=return_period_20,
             ax=ax,
             x_label=label_tz,
             y_label=label_hs,
             line_style='b-',
             sample=sample_struct)
plt.title('Dataset ' + DATASET_CHAR)
plt.show()
