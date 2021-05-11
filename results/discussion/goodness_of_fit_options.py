import numpy as np
import matplotlib.pyplot as plt

from viroconcom.params import ConstantParam, FunctionParam
from viroconcom.distributions import (WeibullDistribution, 
    ExponentiatedWeibullDistribution, MultivariateDistribution)
from viroconcom.read_write import read_ecbenchmark_dataset

MC_FACTOR = 50 # How many more random realizations are drawn than the dataset's length

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

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    F = np.arange(1, n + 1) / n
    return(x, F)


variable_list = [u, hs]
var_symbols = ['$U_{10}$ (m/s)', '$H_s$ (m)']
var_labels = [lu, lhs]

# Use a Monte Carlo sample from the joint model for subsequent calculations
mc_sample = joint_model_4.draw_sample(len(u) * MC_FACTOR)
mc_u = mc_sample[0]
mc_hs = mc_sample[1]

# Probability - Quantile plot
for i, x in enumerate(variable_list):
    x_ordered, F = ecdf(x)
    alpha = 1 - F
    tr = 1 / (alpha * 365.25 * 24)
    tr_threshold = 0.0001
    axs1[i].plot(tr[tr>=tr_threshold], x_ordered[tr>=tr_threshold], 'ok', markerfacecolor='none', ms=3,
     rasterized=True)
    if i == 0:
        model_ordered = joint_model_4.marginal_icdf(F, i)
    else:

        model_ordered, F = ecdf(mc_hs)
        alpha = 1 - F
        tr = 1 / (alpha * 365.25 * 24)
        # Highest F would be 1, which would lead to x_model = inf for unbounded models --> exclude highest value
    mask = (tr >= tr_threshold) & (tr < 50)
    axs1[i].plot(tr[mask], model_ordered[mask], rasterized=True)

    axs1[i].set_xscale('log')
    axs1[i].set_ylabel(var_labels[i].capitalize())
    axs1[i].set_xlabel('Return period (years)')
axs1[0].legend(['Empirical', 'Model'],            
           loc='upper left', 
           ncol=1, 
           prop={'size': 8},
           frameon=False)

# Quantile - Quantile plot
for i, x in enumerate(variable_list):
    x_ordered, F = ecdf(x)
    alpha = 1 - F
    tr = 1 / (alpha * 365.25 * 24)
    if i == 0:
        model_ordered = joint_model_4.marginal_icdf(F, i)
    else:
        model_ordered, F_mc = ecdf(mc_hs) # Use monte carlo sample from probability - quantile plot
        model_ordered = np.quantile(model_ordered, F)
    mask = (tr >= tr_threshold) & (tr < 50)
    axs1[i + 2].plot(model_ordered[mask], x_ordered[mask], 'ok', markerfacecolor='none', ms=3, 
     rasterized=True)
    axs1[i + 2].plot(model_ordered[mask], model_ordered[mask],  rasterized=True)
    axs1[i + 2].set_ylabel(var_symbols[i] + ', ordered values')
    axs1[i + 2].set_xlabel(var_symbols[i] + ', theoretical quantiles')
for ax in axs1:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Joint model plot to analyze the dependence structure
fig2, axs2 = plt.subplots(1, 3, figsize=(12, 4))

# Constant density lines
v_step = 0.1
h_step = 0.1
vgrid, hgrid = np.mgrid[0:40:v_step, 0:20:h_step]
f = np.empty_like(vgrid)
for i in range(vgrid.shape[0]):
    for j in range(vgrid.shape[1]):
        f[i,j] = joint_model_4.pdf([vgrid[i,j], hgrid[i,j]])

density_levels = [1E-6, 1E-5, 1E-4, 1E-3, 1E-2, 1E-1]
axs2[0].scatter(u, hs, s=3, c='black', alpha=0.5, zorder=-2, rasterized=True)
from palettable.colorbrewer.diverging import Spectral_6 as mycorder
level_colors = np.array(mycorder.mpl_colors)
CS = axs2[0].contour(vgrid, hgrid, f, density_levels, colors=level_colors)
axs2[0].set_xlim(0, 35)
axs2[0].set_ylim(0, 16)

lines = [CS.collections[0], CS.collections[1], CS.collections[2], CS.collections[3], 
    CS.collections[4], CS.collections[5]]
labels = ['$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$']
lgd = axs2[0].legend(lines, labels, 
           loc='upper left', 
           ncol=1, 
           prop={'size': 8},
           frameon=False,
           title='Probabilty density')
axs2[0].set_xlabel(lu.capitalize())
axs2[0].set_ylabel(lhs.capitalize())

# Decile curves
u_for_centiles = np.arange(1, 30, 1)
u_bin_width = 2
rv_values = np.array([u_for_centiles, np.empty(len(u_for_centiles))])
quantiles = np.array([0.1, 0.5, 0.9, 0.99])
hs_at_p_empirical = np.empty(len(u_for_centiles))
hs_at_p_model = np.empty(len(u_for_centiles))
from palettable.colorbrewer.diverging import Spectral_4 as mycorder
centile_colors = np.array(mycorder.mpl_colors)

# Empirical decile curves
for i, q in enumerate(quantiles):
    for j, u_bin_center in enumerate(u_for_centiles):
        hs_in_interval = hs[(u > u_bin_center - 0.5 * u_bin_width) & (u < u_bin_center + 0.5 * u_bin_width)]
        hs_at_p_empirical[j] = np.quantile(hs_in_interval, q)

    axs2[1].plot(u_for_centiles, hs_at_p_empirical, label='{:.0f}, empirical'.format(q * 100), 
        linestyle='--', color=centile_colors[i])

# Joint model decile curves
for i, q in enumerate(quantiles):
    p = np.zeros(len(u_for_centiles)) + q
    hs_at_p_model = joint_model_4.distributions[1].i_cdf(p, rv_values=rv_values, 
        dependencies=(0, None, 0, None))
    axs2[1].plot(u_for_centiles, hs_at_p_model, label='{:.0f}, model'.format(q * 100), 
        linestyle='-', color=centile_colors[i])

axs2[1].set_xlabel(lu.capitalize())
axs2[1].set_ylabel(lhs.capitalize())
lgd = axs2[1].legend(loc='upper left', 
           ncol=2, 
           prop={'size': 8},
           frameon=False,
           title='Percentile')

# Upper tail dependence, lambda, plot 
alphas = np.divide(1, np.logspace(2, 5, num=200))
ps = np.subtract(1,  alphas)
lamdas_empirical = np.empty(len(ps))
lamdas_model = np.empty(len(ps))

# Empirical lambda
for i, p in enumerate(ps):
    ur = np.quantile(u, p)
    hr = np.quantile(hs, p)
    mask = [(u > ur) & (hs > hr)]
    jointAB = np.sum(mask) / len(u)
    lamdas_empirical[i] = jointAB / (1 - p)
    
# Joint model lambda
for i, p in enumerate(ps):
    ur = np.quantile(mc_u, p)
    hr = np.quantile(mc_hs, p)
    mask = [(u > ur) & (hs > hr)]
    jointAB = np.sum(mask) / len(u)
    lamdas_model[i] = jointAB / (1 - p)

trs = np.divide(1, np.multiply(alphas,  365.25 * 24))
axs2[2].plot(trs, lamdas_empirical, '--k', label='Empirical')
axs2[2].plot(trs, lamdas_model, '-k', label='Joint model')
axs2[2].set_xscale('log')
axs2[2].set_xlabel('Return period (years)')
axs2[2].set_ylabel('$\hat{\lambda}$ (-)')
lgd = axs2[2].legend(loc='lower left', 
           ncol=1, 
           prop={'size': 8},
           frameon=False)

for ax in axs2:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  

fig1.tight_layout()
fig2.tight_layout()
plt.show()
fig1.savefig('results/discussion/gfx/gof-marginal.pdf',  bbox_inches='tight', pad_inches=0)
fig2.savefig('results/discussion/gfx/gof-joint.pdf',  bbox_inches='tight', pad_inches=0)
