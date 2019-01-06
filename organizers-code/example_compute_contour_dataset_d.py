import numpy as np

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from plot import plot_contour
from contour_statistics import points_outside
from read_write import read_dataset, determine_file_name, write_contour, read_contour

# Read dataset D.
sample_v, sample_hs = read_dataset('../datasets/D.txt')

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

# Compute an IFORM-contour with a return period of 50 years.
return_period = 50
iform_contour = IFormContour(my_fit.mul_var_dist, return_period, 1, 100)

# Save the contour as a csv file in the required format.
file_name = determine_file_name('John', 'Doe', 'D', return_period)
write_contour(iform_contour.coordinates[0][0],
              iform_contour.coordinates[0][1],
              file_name,
              label_x='significant wave height [m]',
              label_y='wind speed [m/s]')
# Read the contour from a csv file in the required format.
(contour_hs, contour_v) = read_contour(file_name)

# Find datapoints that exceed the contour.
hs_outside, v_outside, hs_inside, v_inside = \
    points_outside(contour_hs,
                   contour_v,
                   np.asarray(sample_hs),
                   np.asarray(sample_v))
print('Number of points outside the contour: ' +  str(len(hs_outside)))

# Plot the contour.
sample_struct = [np.asarray(sample_v),
                 np.asarray(sample_hs),
                 v_inside,
                 hs_inside,
                 v_outside,
                 hs_outside,
                 [False, True, False, True]]
plot_contour(contour_v,
             contour_hs,
             return_period,
             'wind speed (m/s)',
             'significant wave height (m)',
             sample_struct)
