import numpy as np

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import (WeibullDistribution, 
    ExponentiatedWeibullDistribution, MultivariateDistribution)
from viroconcom.read_write import read_ecbenchmark_dataset

# Results with n=10, and precision_factor=10: 
#alpha = 1E-2 # lambda contr. 1&2: 0.480 ± 0.007 | contr. 4 = 0.532 ± 0.005 | empirical (p, r): 0.552, 0.851
#alpha = 1E-3 # lambda contr. 1&2: 0.384 ± 0.025 | contr. 4 = 0.403 ± 0.015 | empirical (p, r): 0.571, 0.891
#alpha = 1E-4 # lambda contr. 1&2: 0.334 ± 0.021 | contr. 4 = 0.280 ± 0.020 | empirical (p, r): 0.814, 0.572

alphas = np.array([1E-2, 1E-3, 1E-4])
n_iters = 10 # Number of iterations to estimate the marginal icdf.
precision = 10 # precision_factor to estiamate the marginal icdf (higher = more precise).

# Create joint distribution model of contribution #1 and #2 (baseline model)
u_scale = FunctionParam('power3', 0, 7.58, 0.520)
u_shape = FunctionParam('power3', 0, 3.89, 0.497)
Hs = WeibullDistribution(scale=1.58, shape=1.41, loc=1.02)
U10 = WeibullDistribution(scale=u_scale, shape=u_shape)
distributions = [Hs, U10]
dependencies = [(None, None, None), (0, None, 0)]
joint_model_1 = MultivariateDistribution(distributions, dependencies)

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

joint_models = [joint_model_1, joint_model_4]
model_names = ['Contribution 1 and 2', 'Contribution 4']
u_dim = [1, 0] # Indices of wind speed the different hierarchical joint models.
hs_dim = [0, 1]# Indices of wave height the different hierarchical joint models.

for alpha in alphas:
    print(f'*** Computing with alpha = {alpha:.1e} ***') 
    for j, jmodel in enumerate(joint_models):
        print(f'* Joint model: {model_names[j]} *')    
        lam = np.empty(n_iters)
        for i in range(n_iters):
            ur = jmodel.marginal_icdf(1 - alpha, dim=u_dim[j], precision_factor=precision)
            hr = jmodel.marginal_icdf(1- alpha, dim=hs_dim[j], precision_factor=precision)
            print(f'i = {i + 1}/{n_iters}')
            print(f'u_r = {ur:.3f}, h_r = {hr:.3f}')
            if u_dim[j] == 0:
                p_urhr = jmodel.cdf([ur, hr], lower_integration_limit=(0, 0))
            elif u_dim[j] == 1:
                p_urhr = jmodel.cdf([hr, ur], lower_integration_limit=(0, 0))
            else:
                raise  ValueError
            jointAB = 1 - 2 * (1- alpha) + p_urhr
            lam[i] = jointAB / alpha
            print(f'model lambda[{i + 1:d}] = {lam[i]:.3f}')
            if i == n_iters - 1:
                print(f'model lambda = {np.mean(lam):.3f} ± {np.std(lam):.3f}')

    file_names = ['datasets/D.txt', 'datasets-retained/Dr.txt']
    file_labels = ['D_provided', 'D_retained']
    for i, fname in enumerate(file_names):
        u10, hs, lhs, ltz = read_ecbenchmark_dataset(fname)
        ur = np.quantile(u10, 1 - alpha)
        hr = np.quantile(hs, 1 - alpha)
        jointAB = 1 - 2 * (1 - alpha) + jmodel.cdf([ur, hr], lower_integration_limit=(0, 0))
        lam = jointAB / alpha
        print(f'{file_labels[i]}, empirical lambda = {lam:.3f}')
