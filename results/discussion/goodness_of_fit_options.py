import numpy as np
import matplotlib.pyplot as plt

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import (WeibullDistribution, 
    ExponentiatedWeibullDistribution, MultivariateDistribution)
from viroconcom.read_write import read_ecbenchmark_dataset



# Create joint distribution model of contribution #4.
U10 = ExponentiatedWeibullDistribution(shape=2.42, scale=10.0, shape2=0.761)
hs_shape = FunctionParam('logistics4', 0.582, 1.90, 0.248, 8.49)
hs_scale = FunctionParam('alpha3', 0.394, 0.0178, 1.88,
                            C1=0.582, C2=1.90, C3=0.248, C4=8.49)
hs_shape2 = ConstantParam(5)
Hs = ExponentiatedWeibullDistribution(shape=hs_shape, scale=hs_scale, shape2=hs_shape2)
distributions = [U10, Hs]
dependencies = [(None, None, None, None) , (0, None, 0, None) ]
joint_model_4 = MultivariateDistribution(distributions, dependencies)

joint_models = [joint_model_4]
model_names = ['Contribution 4']
u_dim = [0] # Indices of wind speed the different hierarchical joint models.
hs_dim = [1]# Indices of wave height the different hierarchical joint models.

file_name_provided = 'datasets/D.txt'
file_name_retained = 'datasets-retained/Dr.txt'
u_p, hs_p, lu, lhs = read_ecbenchmark_dataset(file_name_provided)
u_r, hs_r, lu, lhs = read_ecbenchmark_dataset(file_name_retained)
u = np.append(u_p, u_r)
hs = np.append(hs_p, hs_r)


fig1, axs1 = plt.subplots(1, 4, figsize=(12, 3))
fig2, axs2 = plt.subplots(1, 3, figsize=(9, 3))

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    F = np.arange(1, n + 1) / n
    return(x, F)


variable_list = [u, hs]
var_labels = [lu, lhs]
for i, x in enumerate(variable_list):
    x_ordered, F = ecdf(x)
    alpha = 1 - F
    tr = 1 / (alpha * 365.25 * 24)
    axs1[i].plot(tr[tr>=0.001], x_ordered[tr>=0.001], 'ok', markerfacecolor='none', ms=3)
    axs1[i].set_xscale('log')
    axs1[i].set_ylabel(var_labels[i].capitalize())
    axs1[i].set_xlabel('Return period (years)')

plt.show()
