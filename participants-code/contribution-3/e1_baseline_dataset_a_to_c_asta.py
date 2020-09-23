#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:59:12 2020

@author: astah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, weibull_min, chi2, lognorm, kstest
from scipy.optimize import curve_fit
from read_write import determine_file_name_e1, write_contour, read_contour
from contour_statistics import points_outside
from plot import PlottedSample, plot_contour
from read_write import read_dataset
plt.close('all')

plt.close('all')
#%% Functions to fit
# Power function
def power3(x, a, b, c):
    return a + b * x ** c

# Exponential function
def exp3(x, a, b, c):
    return a + b * np.exp(c * x)

#%% Read dataset A, B  or C.
DATASET_CHAR = 'A'
file_path = '../datasets/' + DATASET_CHAR + '.txt'
sample_hs, sample_tz, label_hs, label_tz= read_dataset(file_path)

df = pd.read_csv(file_path, sep='; ')
#%% Inspect the marginal distributions

weib_par1 = weibull_min.fit(df[df.columns[1]], loc=0)
logn_par1 = lognorm.fit(df[df.columns[1]], loc=0)

weib_par2 = weibull_min.fit(df[df.columns[2]], loc=0)
logn_par2 = lognorm.fit(df[df.columns[2]], loc=0)

#%% Goodness of fit

print(kstest(df[df.columns[1]].values, 'weibull_min', args=weib_par1)) 
print(kstest(df[df.columns[1]].values, 'lognorm', args=logn_par1))

print(kstest(df[df.columns[2]].values, 'weibull_min', args=weib_par2))
print(kstest(df[df.columns[2]].values, 'lognorm', args=logn_par2))

#%% Plot the distributions
#n_bins = 100

#plt.figure()
#plt.subplot(211)
#n1, bins1, _ = plt.hist(df[df.columns[1]], n_bins, density=True, label = df.columns[1])
#plt.plot(bins1, weibull_min.pdf(bins1,*weib_par1), label='Weibull')
#plt.plot(bins1, lognorm.pdf(bins1,*logn_par1), label='Lognorm')
#plt.legend(loc='best')
#plt.subplot(212)
#n, bins, _ = plt.hist(df[df.columns[2]], n_bins, density=True, label = df.columns[2])
#plt.plot(bins, weibull_min.pdf(bins,*weib_par2), label='Weibull')
#plt.plot(bins, lognorm.pdf(bins,*logn_par2), label='Lognorm')
#plt.legend(loc='best')

#%% Bin the data to find the conditoinal marginal distribution
s_min = df[df.columns[1]].min()
s_max = df[df.columns[1]].max()

bin_size = 0.5
s_bins = np.arange(np.floor(s_min), np.ceil(s_max), bin_size) + bin_size/2
s_binedges = s_bins + bin_size/2

s_ind_bin = np.digitize(df[df.columns[1]], bins=s_binedges)

unique, counts = np.unique(s_ind_bin, return_counts=True)

ind_min_bin = unique[counts>10][0]
ind_max_bin = unique[counts>10][-1]
x_bins = s_bins[ind_min_bin:ind_max_bin+1]
real_bins = np.zeros(len(x_bins))

logn_par_cond = np.zeros((len(x_bins),3))
mu_cond = np.zeros(len(x_bins))
sig_cond = np.zeros(len(x_bins))

plot_bins = np.arange(0,14,0.2)

for i in range(len(x_bins)):
    mask1 = s_ind_bin == i + ind_min_bin
    real_bins[i] = df[df.columns[1]][mask1].mean()
    logn_par_cond[i,:] = lognorm.fit(df[df.columns[2]][mask1], floc=0)
    mu_cond[i] = np.mean(np.log(df[df.columns[2]][mask1]))
    sig_cond[i] = np.std(np.log(df[df.columns[2]][mask1]))
#    plt.figure()
#    b = plt.hist(df[df.columns[2]][mask1], bins= plot_bins, density=True)
#    plt.plot(b[1], lognorm.pdf(b[1],*logn_par_cond[i,:]), color='g')

#bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
bounds = ([-1, 0, -np.inf], [np.inf, np.inf, np.inf])
p0_mu = [0, 2, 0.1]
p0_sig = [0.1, 0.1, -0.3]

mu_vars = curve_fit(power3, real_bins, mu_cond, p0=p0_mu, bounds=bounds)[0]
sig_vars = curve_fit(exp3, real_bins, sig_cond, p0=p0_sig, bounds=bounds)[0]

sig_func = curve_fit(exp3, real_bins, logn_par_cond[:,0], p0=p0_sig, bounds=bounds)[0]
mu_func = curve_fit(power3, real_bins, np.log(logn_par_cond[:,2]), p0=p0_mu, bounds=bounds)[0]

plt.figure()
plt.subplot(211)
plt.plot(real_bins, np.log(logn_par_cond[:,2]), 'o')
plt.plot(real_bins, mu_cond, 'o')
plt.plot(x_bins, power3(x_bins, *mu_func))
plt.ylabel(r'$\mu$: scale parameter')
plt.subplot(212)
plt.plot(real_bins, logn_par_cond[:,0], 'o')
plt.plot(real_bins, sig_cond, 'o')
plt.plot(x_bins, exp3(x_bins, *sig_func))
plt.plot(x_bins, exp3(x_bins, *sig_vars))
plt.ylabel(r'$\sigma$: shape parameter')


#%% Perform the IDS

T1 = 1
T20 = 20

#beta1 = norm.ppf(1- 10/(T1*len(df)))
#beta20 = norm.ppf(1- 10/(T20*len(df)))
beta1 = np.sqrt(chi2.ppf(1- 10/(T1*len(df)), df=2))
beta20 = np.sqrt(chi2.ppf(1- 10/(T20*len(df)), df=2))

phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)

u0_1 = beta1*np.cos(phi)
u1_1 = beta1*np.sin(phi)

u0_20 = beta20*np.cos(phi)
u1_20 = beta20*np.sin(phi)

x1_1 = lognorm.ppf( norm.cdf(u1_1), *logn_par1)
x1_20 = lognorm.ppf( norm.cdf(u1_20), *logn_par1)

# The weibull conditional distribution
sig_x1_1 =  exp3(x1_1, *sig_func)
mu_x1_1 =  power3(x1_1, *mu_func)

sig_x1_20 =  exp3(x1_20, *sig_func)
mu_x1_20 =  power3(x1_20, *mu_func)

x0_1 = lognorm.ppf( norm.cdf(u0_1), sig_x1_1, loc=0, scale=np.exp(mu_x1_1))
x0_20 = lognorm.ppf( norm.cdf(u0_20), sig_x1_20, loc=0, scale=np.exp(mu_x1_20))
#%%
h = sns.jointplot(x= df.columns[2] , y=df.columns[1] , data=df, s=5)
h.x, h.y = x0_1, x1_1
h.plot_joint(plt.plot, color='C1')
h.x, h.y = x0_20, x1_20
h.plot_joint(plt.plot, color='C2')

#%% E1 requirements:
# Save the contours as csv files in the required format.
folder_name = 'contour_coordinates/'
file_name_1 = determine_file_name_e1('Asta', 'Hannesdottir', DATASET_CHAR, T1)
write_contour(x1_1, #y-axis
              x0_1,
              folder_name + file_name_1,
              label_x=df.columns[1],
              label_y=df.columns[2])
file_name_20 = determine_file_name_e1('Asta', 'Hannesdottir', DATASET_CHAR, T20)
write_contour(x1_20,
              x0_20,
              folder_name + file_name_20,
              label_x=df.columns[1],
              label_y=df.columns[2])

# Read the contours from the csv files.
(contour_hs_1, contour_tz_1) = read_contour(folder_name + file_name_1)
(contour_hs_20, contour_tz_20) = read_contour(folder_name + file_name_20)

# Find datapoints that exceed the 20-yr contour.
hs_outside, tz_outside, hs_inside, tz_inside = \
    points_outside(contour_hs_20,
                   contour_tz_20,
                   np.asarray(df[df.columns[1]].values),
                   np.asarray(df[df.columns[2]].values))
print('Number of points outside the contour: ' +  str(len(hs_outside)))
#%%
nan_mask = np.isnan(contour_tz_20)

fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)

plotted_sample = PlottedSample(x=np.asarray(sample_tz),
                               y=np.asarray(sample_hs),
                               ax=ax,
                               x_inside=tz_inside,
                               y_inside=hs_inside,
                               x_outside=tz_outside,
                               y_outside=hs_outside,
                               return_period=T20)
# Plot the 1-year contour.
plot_contour(x=contour_tz_1,
             y=contour_hs_1,
             ax=ax,
             contour_label=str(T1) + '-yr contour',
             x_label=label_tz,
             y_label=label_hs,
             line_style='b--',
             plotted_sample=plotted_sample)

# Plot the 20-year contour and the sample.
plot_contour(x=contour_tz_20[~nan_mask],
             y=contour_hs_20[~nan_mask],
             ax=ax,
             contour_label=str(T20) + '-yr contour',
             x_label=label_tz,
             y_label=label_hs,
             line_style='b-')#,
#             plotted_sample=plotted_sample)
plt.title('Dataset ' + DATASET_CHAR)
plt.show()
plt.savefig('../results/figures/hannesdottir_asta_dataset_'+DATASET_CHAR+'_1_20.png', dpi=300)