import numpy as np
import matplotlib.pyplot as plt

from viroconcom.read_write import read_ecbenchmark_dataset
from viroconcom.fitting import Fit
from viroconcom.contours import HighestDensityContour, \
    sort_points_to_form_continous_line
from viroconcom.plot import plot_contour, SamplePlotData

np.random.seed(9001) # For reproducablity.

# Define the number of years of data that one bootstrap sample should contain.
# In the benchmark 1, 5 and 25 years are used.
NR_OF_YEARS_TO_DRAW = [1, 5]
NR_OF_BOOTSTRAP_SAMPLES = [25, 5]
GRID_CELL_SIZE = 0.5 # 0.05

# Read dataset D.
file_path = 'datasets/D.txt'
dataset_d_v, dataset_d_hs, label_v, label_hs = read_ecbenchmark_dataset(file_path)

# Define the origin (will be used to compute confidence intervals).
v0 = np.mean(dataset_d_v)
hs0 = np.mean(dataset_d_hs)

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
# Fit the model to the dataset.
fit = Fit((dataset_d_v, dataset_d_hs), (dist_description_v, dist_description_hs))
dist0 = fit.mul_var_dist.distributions[0]
dist1 = fit.mul_var_dist.distributions[1]

# Compute 50-yr contour.
return_period = 50
ts = 1  # Sea state duration in hours.
limits = [(0, 45), (0, 25)]  # Limits of the computational domain.
deltas = [GRID_CELL_SIZE, GRID_CELL_SIZE]  # Dimensions of the grid cells.
hdc = HighestDensityContour(fit.mul_var_dist, return_period, ts,
                                      limits, deltas)
contour_with_all_data = sort_points_to_form_continous_line(
    hdc.coordinates[0], hdc.coordinates[1], do_search_for_optimal_start=True)

# Create the figure for plotting the contours.
fig, axs = plt.subplots(len(NR_OF_YEARS_TO_DRAW), 2, sharex='row', sharey='row',
                        figsize=(8, 6.5))

for j in range(len(NR_OF_YEARS_TO_DRAW)):
    nr_of_datapoints_to_draw = round(NR_OF_YEARS_TO_DRAW[j] * 365.25 * 24)
    for i in range(NR_OF_BOOTSTRAP_SAMPLES[j]):
        print('j = {}/{}, contour {}/{}'.format(j, len(NR_OF_YEARS_TO_DRAW) - 1,
                                                i + 1, NR_OF_BOOTSTRAP_SAMPLES[j]))

        # Resample from the hindcast dataset to get the sample D_i.
        sample_indices = np.random.randint(dataset_d_v.size, size=nr_of_datapoints_to_draw)
        v_i = np.take(dataset_d_v, sample_indices)
        hs_i = np.take(dataset_d_hs, sample_indices)

        # Fit the model to the dataset.
        fit = Fit((v_i, hs_i), (dist_description_v, dist_description_hs))
        dist0 = fit.mul_var_dist.distributions[0]
        dist1 = fit.mul_var_dist.distributions[1]

        # Compute 50-yr contour.
        hdc_contour_i = HighestDensityContour(fit.mul_var_dist, return_period, ts,
                                              limits, deltas)
        c = sort_points_to_form_continous_line(hdc_contour_i.coordinates[0],
                                               hdc_contour_i.coordinates[1],
                                               do_search_for_optimal_start=True)
        hdc_contour_i.c = c

        if i == 0:
            contours = [hdc_contour_i]
        else:
            contours.append(hdc_contour_i)

    # Plot the environmental contours.
    for i, contour in enumerate(contours):
        if i == 0:
            plotted_sample = SamplePlotData(x=np.asarray(dataset_d_v),
                                           y=np.asarray(dataset_d_hs),
                                           ax=axs)
            contour_label = str(return_period) + '-yr contour'
            plot_contour(x=contour.c[0],
                         y=contour.c[1],
                         ax=axs[j, 0],
                         x_label=label_v.capitalize(),
                         y_label=label_hs.capitalize(),
                         style='b-',
                         alpha=0.4,
                         sample_plot_data=plotted_sample)
        else:
            plot_contour(x=contour.c[0],
                         y=contour.c[1],
                         style='b-',
                         alpha=0.4,
                         ax=axs[j, 0])
    plot_contour(x=contour_with_all_data[0], y=contour_with_all_data[1],
                 color='r', linewidth=2, linestyle='--', ax=axs[j, 0])
    axs[j, 0].set_xlim((0, 40))
    axs[j, 0].set_ylim((0, 25))

    for i in range(NR_OF_BOOTSTRAP_SAMPLES[j]):
        print('j = {}/{}, contour {}/{}'.format(j, len(NR_OF_YEARS_TO_DRAW) - 1,
                                                i + 1, NR_OF_BOOTSTRAP_SAMPLES[j]))
        n_consecutive_points = round(NR_OF_YEARS_TO_DRAW[j] * 365.25 * 24)
        v_i = dataset_d_v[i * n_consecutive_points : (i + 1) * n_consecutive_points]
        hs_i = dataset_d_hs[i * n_consecutive_points: (i + 1) * n_consecutive_points]

        # Fit the model to the dataset.
        fit = Fit((v_i, hs_i), (dist_description_v, dist_description_hs))
        dist0 = fit.mul_var_dist.distributions[0]
        dist1 = fit.mul_var_dist.distributions[1]

        # Compute 50-yr contour.
        return_period = 50
        ts = 1  # Sea state duration in hours.
        limits = [(0, 45), (0, 25)]  # Limits of the computational domain.
        deltas = [GRID_CELL_SIZE, GRID_CELL_SIZE]  # Dimensions of the grid cells.
        hdc_contour_i = HighestDensityContour(fit.mul_var_dist, return_period, ts,
                                              limits, deltas)
        c = sort_points_to_form_continous_line(hdc_contour_i.coordinates[0],
                                               hdc_contour_i.coordinates[1],
                                               do_search_for_optimal_start=True)
        hdc_contour_i.c = c

        if i == 0:
            contours = [hdc_contour_i]
        else:
            contours.append(hdc_contour_i)

    for i, contour in enumerate(contours):
        if i == 0:
            contour_label = str(return_period) + '-yr contour'
            plot_contour(x=contour.c[0],
                         y=contour.c[1],
                         ax=axs[j, 1],
                         x_label=label_v.capitalize(),
                         y_label=label_hs.capitalize(),
                         style='b-',
                         alpha=0.4,
                         sample_plot_data=plotted_sample)
        else:
            plot_contour(x=contour.c[0],
                         y=contour.c[1],
                         style='b-',
                         alpha=0.4,
                         ax=axs[j, 1])
    plot_contour(x=contour_with_all_data[0], y=contour_with_all_data[1],
                 color='r', linewidth=2, linestyle='--', ax=axs[j, 1])
    axs[j, 1].set_xlim((0, 40))
    axs[j, 1].set_ylim((0, 25))

fig.tight_layout()
plt.show()
fig.savefig('results/discussion/e2_autocorrelation_4panels.pdf')
