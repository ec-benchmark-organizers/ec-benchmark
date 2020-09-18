import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from plot import plot_marginal_fit
from read_write import read_dataset

# Read dataset D, E  or F.
DATASET_CHAR = 'F'
file_path = 'datasets/' + DATASET_CHAR + '.txt'
sample_v, sample_hs, label_v, label_hs = read_dataset(file_path)

# Define the structure of the probabilistic model that will be fitted to the
# dataset.
dist_description_v = {'name': 'Weibull_2p',
                      'dependency': (None, None, None),
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

fig = plt.figure(figsize=(3.5, 3.3), dpi=150)
plot_marginal_fit(sample_v, dist0, fig=fig, label='$v$ (m s$^{-1}$)',
                  dataset_char=DATASET_CHAR)
plt.tight_layout()
plt.show()
