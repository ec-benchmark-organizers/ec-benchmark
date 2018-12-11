from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from plot import plot_contour
import csv

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
dist_description_hs = {'name': 'Weibull',
                      'dependency': (None, None, None),
                      'width_of_intervals': 0.5}
dist_description_tz = {'name': 'Lognormal',
                      'dependency': (0,  None, 0), #Shape, Location, Scale
                      'functions': ('power3', None, 'exp3') #Shape, Location, Scale
                      }

# Fit the model to the data.
my_fit = Fit((sample_hs, sample_tz), (dist_description_hs, dist_description_tz))

# Compute an IFORM-contour with a return period of 50 years.
iform_contour = IFormContour(my_fit.mul_var_dist, 50, 1, 100)

# Plot the contour.
plot_contour(iform_contour.coordinates[0][1],
             iform_contour.coordinates[0][0],
             50,
             'zero-up-crossing period (s)',
             'significant wave height (m)',
             [sample_tz, sample_hs])
