#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:19:43 2020

@author: astah
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, weibull_min, chi2, lognorm
from scipy.optimize import curve_fit
from read_write import determine_file_name_e1, write_contour, read_contour
from contour_statistics import points_outside
from plot import PlottedSample, plot_contour
from read_write import read_dataset
plt.close('all')

#%% Functions to fit
# Power function
def power3(x, a, b, c):
    return a + b * x ** c

#%% Read dataset D, E or F.
DATASET_CHAR = 'F'
file_path = '../datasets/' + DATASET_CHAR + '.txt'
sample_v, sample_hs, label_v, label_hs = read_dataset(file_path)

df = pd.read_csv(file_path, sep='; ')
#%% Inspect the marginal distributions

weib_par1 = weibull_min.fit(df[df.columns[1]], loc=0)
logn_par1 = lognorm.fit(df[df.columns[1]], loc=0)

weib_par2 = weibull_min.fit(df[df.columns[2]], loc=0)
logn_par2 = lognorm.fit(df[df.columns[2]], loc=0)

#%% Plot the distributions
#n_bins = 100
#
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
s_min = df[df.columns[2]].min()
s_max = df[df.columns[2]].max()

bin_size = 0.5
s_bins = np.arange(np.floor(s_min), np.ceil(s_max), bin_size) + bin_size/2
s_binedges = s_bins + bin_size/2

s_ind_bin = np.digitize(df[df.columns[2]], bins=s_binedges)

unique, counts = np.unique(s_ind_bin, return_counts=True)

ind_min_bin = unique[counts>10][0]
ind_max_bin = unique[counts>10][-1]
x_bins = s_bins[ind_min_bin:ind_max_bin+1]
real_bins = np.zeros(len(x_bins))

weib_par_cond = np.zeros((len(x_bins),3))

plot_bins = np.arange(0,30,0.5)

for i in range(len(x_bins)):
    mask1 = s_ind_bin == i + ind_min_bin
    real_bins[i] = df[df.columns[2]][mask1].mean()
    weib_par_cond[i,:] = weibull_min.fit(df[df.columns[1]][mask1], floc=0)
#    plt.figure()
#    b = plt.hist(df[df.columns[1]][mask1], bins= plot_bins, density=True)
#    plt.plot(b[1],weibull_min.pdf(b[1],*weib_par_cond[i,:]), color='g')

#mask_good = weib_par_cond[:,1] > 0


bounds = ([0,0,0],[np.inf, np.inf, np.inf])

k = curve_fit(power3, real_bins, weib_par_cond[:,0], bounds=bounds)[0]
a = curve_fit(power3, real_bins, weib_par_cond[:,2], bounds=bounds)[0]

plt.figure()
plt.subplot(211)
plt.plot(real_bins, weib_par_cond[:,0], 'o')
plt.plot(x_bins, power3(x_bins, *k))
plt.ylabel('k: shape param')
plt.subplot(212)
plt.plot(real_bins, weib_par_cond[:,2], 'o')
plt.plot(x_bins, power3(x_bins, *a))
plt.ylabel('A: scale param')

#%% Perform the IDS

T1 = 1
T50 = 50

#beta1 = norm.ppf(1- 10/(T1*len(df)))
beta1 = np.sqrt(chi2.ppf(1- 25/(T1*len(df)), df=2))
beta50 = np.sqrt(chi2.ppf(1- 25/(T50*len(df)), df=2))

phi = np.linspace(0, 2 * np.pi, 360, endpoint=False)

u0_1 = beta1*np.cos(phi)
u1_1 = beta1*np.sin(phi)

u0_50 = beta50*np.cos(phi)
u1_50 = beta50*np.sin(phi)

x1_1 = weibull_min.ppf( norm.cdf(u1_1), *weib_par2)
x1_50 = weibull_min.ppf( norm.cdf(u1_50), *weib_par2)

# The weibull conditional distribution
k_x1_1 =  power3(x1_1, *k)
a_x1_1 =  power3(x1_1, *a)

k_x1_50 =  power3(x1_50, *k)
a_x1_50 =  power3(x1_50, *a)

x0_1 = weibull_min.ppf( norm.cdf(u0_1), k_x1_1, loc=0, scale=a_x1_1)
x0_50 = weibull_min.ppf( norm.cdf(u0_50), k_x1_50, loc=0, scale=a_x1_50)
#%%
h = sns.jointplot(x= df.columns[1] , y=df.columns[2] , data=df, s=5)
h.x, h.y = x0_1, x1_1
h.plot_joint(plt.plot, color='C1')
h.x, h.y = x0_50, x1_50
h.plot_joint(plt.plot, color='C2')

#%% E1 requirements:
# Save the contours as csv files in the required format.
folder_name = 'contour_coordinates/'
file_name_1 = determine_file_name_e1('Asta', 'Hannesdottir', DATASET_CHAR, T1)
write_contour(x1_1, #y-axis
              x0_1,
              folder_name + file_name_1,
              label_x=df.columns[2],
              label_y=df.columns[1])
file_name_50 = determine_file_name_e1('Asta', 'Hannesdottir', DATASET_CHAR, T50)
write_contour(x1_50,
              x0_50,
              folder_name + file_name_50,
              label_x=df.columns[2],
              label_y=df.columns[1])

# Read the contours from the csv files.
(contour_hs_1, contour_v_1) = read_contour(folder_name + file_name_1)
(contour_hs_50, contour_v_50) = read_contour(folder_name + file_name_50)

# Find datapoints that exceed the 20-yr contour.
hs_outside, v_outside, hs_inside, v_inside = \
    points_outside(contour_hs_50,
                   contour_v_50,
                   np.asarray(df[df.columns[2]].values),
                   np.asarray(df[df.columns[1]].values))
print('Number of points outside the contour: ' +  str(len(hs_outside)))

#%%
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)

# Plot the 1-year contour.
plot_contour(x=contour_v_1,
             y=contour_hs_1,
             ax=ax,
             contour_label=str(T1) + '-yr contour',
             x_label=label_v,
             y_label=label_hs,
             line_style='b--')

# Plot the 50-year contour and the sample.
plotted_sample = PlottedSample(x=np.asarray(sample_v),
                               y=np.asarray(sample_hs),
                               ax=ax,
                               x_inside=v_inside,
                               y_inside=hs_inside,
                               x_outside=v_outside,
                               y_outside=hs_outside,
                               return_period=T50)
plot_contour(x=contour_v_50,
             y=contour_hs_50,
             ax=ax,
             contour_label=str(T50) + '-yr contour',
             x_label=label_v,
             y_label=label_hs,
             line_style='b-',
             plotted_sample=plotted_sample)
plt.title('Dataset ' + DATASET_CHAR)
plt.show()
plt.savefig('../results/figures/hannesdottir_asta_dataset_'+DATASET_CHAR+'_1_50.png', dpi=300)