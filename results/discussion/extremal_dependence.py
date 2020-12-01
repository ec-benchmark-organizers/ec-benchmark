import numpy as np

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import (WeibullDistribution, 
    ExponentiatedWeibullDistribution, MultivariateDistribution)
from viroconcom.read_write import read_ecbenchmark_dataset

# Results: 
#u = 1 - 1E-2 # lambda contr. 1 & 2: 0.481, 0.475, 0.461| contr. 4 = 0.526, 0.526, 0.524 | empirical (p, r): 0.552, 0.851
#u = 1 - 1E-3 # lambda contr. 1 & 2: 0.405 | contr. 4 = 0.415, 0.405, 0.406 | empirical (p, r): 0.571, 0.891
#u = 1 - 1E-4 # lambda contr. 1 & 2: 0.378 | contr. 4 = 0.292, 0.291, 0.302 | empirical (p, r): 0.814, 0.572
#u = 1 - 1E-5 # lambda contr. 1 & 2: X.XXX | contr. 4 = 0.215, 0.211, 0.213 | empirical (p, r): -1.119

us = 1 - np.array([1E-2, 1E-3, 1E-4])

# Create joint distribution model of contribution #1 and #2 (baseline model)
alpha = FunctionParam('power3', 0, 7.58, 0.520)
beta = FunctionParam('power3', 0, 3.89, 0.497)
Hs = WeibullDistribution(scale=1.58, shape=1.41, loc=1.02)
U10 = WeibullDistribution(scale=alpha, shape=beta)
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

for u in us:
    print(f'*** Computing with u = 1 - {1-u:.1e} ***') 
    for j, joint_model in enumerate(joint_models):
        print(f'* Joint model: {model_names[j]} *')    
        n_iters = 3
        for i in range(n_iters):
            ur = joint_model.marginal_icdf(u, dim=0)
            hr = joint_model.marginal_icdf(u, dim=1, precision_factor=5)
            print(f'i = {i+1}/{n_iters}')
            print(f'u_r = {ur:.3f}, h_r = {hr:.3f}')
            jointAB = 1 - 2 * u + joint_model.cdf([ur, hr], lower_integration_limit=(0, 0))
            lam = jointAB / (1 - u)
            print(f'model lambda = {lam:.3f}')

    file_names = ['datasets/D.txt', 'datasets-retained/Dr.txt']
    file_labels = ['D_provided', 'D_retained']
    for i, fname in enumerate(file_names):
        u10, hs, lhs, ltz = read_ecbenchmark_dataset(fname)
        ur = np.quantile(u10, u)
        hr = np.quantile(hs, u)
        jointAB = 1 - 2 * u + joint_model.cdf([ur, hr], lower_integration_limit=(0, 0))
        lam = jointAB / (1 - u)
        print(f'{file_labels[i]}, empircal lambda = {lam:.3f}')