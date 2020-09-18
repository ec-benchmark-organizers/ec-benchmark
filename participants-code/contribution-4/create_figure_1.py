import numpy as np
import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from statistics import median

from plot import plot_contour, PlottedSample
from read_write import read_dataset

# Read dataset A
DATASET_CHAR = 'A'
file_path = 'datasets/' + DATASET_CHAR + '.txt'
a_hs, a_tz, label_hs, label_tz= read_dataset(file_path)

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

# Fit the hs-tz model to the data.
fit = Fit((a_hs, a_tz), (dist_description_hs, dist_description_tz))
dist0 = fit.mul_var_dist.distributions[0]

# Compute IFORM-contour with return periods of 20 years.
return_period_20 = 20
iform_contour_20 = IFormContour(fit.mul_var_dist, return_period_20, 1, 100)
contour_hs_20 = iform_contour_20.coordinates[0][0]
contour_tz_20 = iform_contour_20.coordinates[0][1]

# Read dataset D.
DATASET_CHAR = 'D'
file_path = 'datasets/' + DATASET_CHAR + '.txt'
d_v, d_hs, label_v, label_hs= read_dataset(file_path)

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
fit = Fit((d_hs, d_v), (dist_description_hs, dist_description_v))
dist0 = fit.mul_var_dist.distributions[0]
dist1 = fit.mul_var_dist.distributions[1]

# Compute IFORM-contours with return periods of 50 years.
return_period_50 = 50
iform_contour_50 = IFormContour(fit.mul_var_dist, return_period_50, 1, 100)
contour_hs_50 = iform_contour_50.coordinates[0][0]
contour_v_50 = iform_contour_50.coordinates[0][1]

# Compute the binned median and start by defining intervals.
MIN_DATA_POINTS_FOR_FIT = 10
interval_width = 0.5
interval_centers = np.arange(
    0.5 * interval_width,
    max(d_v) + 0.5 * interval_width,
    interval_width)
# Sort sample.
samples = np.stack((d_hs, d_v)).T
sort_indice = np.argsort(samples[:, 1])
sorted_samples = samples[sort_indice]
# Initialize variables
medians = []
deleted_centers = []
# Define the data interval that is used for the fit.
for i, step in enumerate(interval_centers):
    mask = ((sorted_samples[:, 1] >= step - 0.5 * interval_width) &
            (sorted_samples[:, 1] < step + 0.5 * interval_width))
    samples_in_interval = sorted_samples[mask, 0]
    if len(samples_in_interval) >= MIN_DATA_POINTS_FOR_FIT:
        medians.append(median(samples_in_interval))

    else:
        # For case that too few fitting data for the step were found
        # the step is deleted.
        deleted_centers.append(i)  # Add index of unused center.
# Delete interval centers that were not used.
interval_centers = np.delete(interval_centers, deleted_centers)


# Create Figure 1 of the paper.
fig = plt.figure(figsize=(10, 5), dpi=150)

# Plot dataset A and the 20-year contour.
ax = fig.add_subplot(121)
plotted_sample_a = PlottedSample(x=np.asarray(a_tz),
                               y=np.asarray(a_hs),
                               ax=ax,
                               return_period=return_period_20)
plot_contour(x=contour_tz_20,
             y=contour_hs_20,
             ax=ax,
             contour_label=str(return_period_20) + '-yr contour',
             x_label=label_tz,
             y_label=label_hs,
             line_style='b-',
             plotted_sample=plotted_sample_a)
plt.legend(['20-year contour', '10 years of observations'], loc='upper left',
           frameon=False)
plt.title('Dataset A')

# Plot dataset D and the 50-year contour.
ax2 = fig.add_subplot(122)
plotted_sample_d = PlottedSample(x=np.asarray(d_v),
                               y=np.asarray(d_hs),
                               ax=ax2,
                               return_period=return_period_50)
plot_contour(x=contour_v_50,
             y=contour_hs_50,
             ax=ax2,
             contour_label=str(return_period_50) + '-yr contour',
             x_label=label_v,
             y_label=label_hs,
             line_style='b-',
             plotted_sample=plotted_sample_d)
plt.legend(['50-year contour', '25 years of observations', ], loc='upper left',
           frameon=False)
plt.title('Dataset D')
plt.show()

