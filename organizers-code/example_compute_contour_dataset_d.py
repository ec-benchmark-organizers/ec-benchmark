import numpy as np
import csv

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from plot import plot_contour
from contour_statistics import points_outside


sample_hs = list()
sample_v = list()
with open('../datasets/D.txt', newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=';')
    idx = 0
    for row in reader:
        if idx > 0: # Ignore the header
            sample_v.append(float(row[1]))
            sample_hs.append(float(row[2]))
        idx = idx + 1

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

# Find datapoints that exceed the contour.
hs_outside, v_outside, hs_inside, v_inside = \
    points_outside(iform_contour.coordinates[0][0],
                   iform_contour.coordinates[0][1],
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
plot_contour(iform_contour.coordinates[0][1],
             iform_contour.coordinates[0][0],
             return_period,
             'wind speed (m/s)',
             'significant wave height (m)',
             sample_struct)
