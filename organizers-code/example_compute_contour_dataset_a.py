import numpy as np
import csv

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from plot import plot_contour
from contour_statistics import points_outside

sample_hs = list()
sample_tz = list()
with open('../datasets/A.txt', newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=';')
    idx = 0
    for row in reader:
        if idx > 0: # Ignore the header
            sample_hs.append(float(row[1]))
            sample_tz.append(float(row[2]))
        idx = idx + 1

# Define the structure of the probabilistic model that will be fitted to the
# dataset. We will use the model that is recommended in DNV-RP-C205 (2010) on
# page 38 and that is called 'conditonal modeling approach' (CMA).
dist_description_hs = {'name': 'Weibull_3p',
                      'dependency': (None, None, None),
                      'width_of_intervals': 0.5}
dist_description_tz = {'name': 'Lognormal',
                      'dependency': (0,  None, 0), #Shape, Location, Scale
                      'functions': ('power3', None, 'exp3') #Shape, Location, Scale
                      }

# Fit the model to the data.
my_fit = Fit((sample_hs, sample_tz), (dist_description_hs, dist_description_tz))

# Compute an IFORM-contour with a return period of 20 years.
return_period = 20
iform_contour = IFormContour(my_fit.mul_var_dist, return_period, 1, 100)

# Find datapoints that exceed the contour.
hs_outside, tz_outside, hs_inside, tz_inside = \
    points_outside(iform_contour.coordinates[0][0],
                   iform_contour.coordinates[0][1],
                   np.asarray(sample_hs),
                   np.asarray(sample_tz))
print('Number of points outside the contour: ' +  str(len(hs_outside)))

# Plot the contour.
sample_struct = [np.asarray(sample_tz),
                 np.asarray(sample_hs),
                 tz_inside,
                 hs_inside,
                 tz_outside,
                 hs_outside,
                 [True, True, False, True]]
plot_contour(iform_contour.coordinates[0][1],
             iform_contour.coordinates[0][0],
             return_period,
             'zero-up-crossing period (s)',
             'significant wave height (m)',
             sample_struct)
