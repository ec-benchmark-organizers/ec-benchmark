#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:13:16 2020

@author: astah
"""

import numpy as np
import matplotlib.pyplot as plt

from viroconcom.fitting import Fit
from viroconcom.contours import IFormContour
from plot import plot_contour, PlottedSample, plot_confidence_interval
from read_write import read_dataset, determine_file_name_e2, write_contour, read_contour
from contour_intersection import contour_intersection
from contour_statistics import thetastar_to_theta
from scipy.stats import norm, weibull_min, chi2
from scipy.optimize import curve_fit

# Define the number of years of data that one bootstrap sample should contain.
# In the exercise 1, 5 and 25 years are used.
NR_OF_YEARS_TO_DRAW = 25

DO_COMPUTE_CONFIDENCE_INTERVAL = True
NR_OF_BOOTSTRAP_SAMPLES = 1000 # Must be 1000 in Exercise 2.
BOTTOM_PERCENTILE = 2.5 # Must be 2.5 in Exercise 2.
UPPER_PERCENTILE = 97.5 # Must be 97.5 in Excercise 2.
ANGLE_STEP_FOR_CI = 2 # Must be 2 in in Excercise 2.

DO_PLOT_ANGLE_LINES = False # Must be False in Excercise 2. For visualization.
NR_OF_POINTS_ON_CONTOUR = 200 # For IFORM contours it can be set explicitly.

# Read dataset D.
file_path = '../datasets/D.txt'
dataset_d_v, dataset_d_hs, label_v, label_hs = read_dataset(file_path)

# Define the origin (will be used to compute confidence intervals).
v0 = np.mean(dataset_d_v)
hs0 = np.mean(dataset_d_hs)
#print('Origin:')
#print(v0)
#print(hs0)

nr_of_datapoints_to_draw = int(NR_OF_YEARS_TO_DRAW * 365.25 * 24)
return_period = 50
p_r = 1/int(return_period*365.25*24)
#np.random.seed(9001)
bin_size = 0.5
bounds = ([0,0,0],[np.inf, np.inf, np.inf])
phi = np.linspace(0, 2 * np.pi, NR_OF_POINTS_ON_CONTOUR)
beta50 = np.sqrt(chi2.ppf(1-p_r, df=2))
u0_50 = beta50*np.cos(phi)
u1_50 = beta50*np.sin(phi)

theta_v_ij = np.zeros(shape=(NR_OF_BOOTSTRAP_SAMPLES, int(360/ANGLE_STEP_FOR_CI)))
theta_hs_ij = np.zeros(shape=(NR_OF_BOOTSTRAP_SAMPLES, int(360/ANGLE_STEP_FOR_CI)))


def power3(x, a, b, c):
    return a + b * x ** c

for i in range(NR_OF_BOOTSTRAP_SAMPLES):
    # Resample from the hindcast dataset to get the sample D_i.
    sample_indices = np.random.randint(dataset_d_v.size, size=nr_of_datapoints_to_draw)
    v_i = np.take(dataset_d_v, sample_indices)
    hs_i = np.take(dataset_d_hs, sample_indices)
    # Fit Weibull to Hs:
    weib_par2 = weibull_min.fit(hs_i, loc=0)
    # Find the conditional Weibull for V:
    h_min = hs_i.min()
    h_max = hs_i.max()

    h_bins = np.arange(np.floor(h_min), np.ceil(h_max), bin_size) + bin_size/2
    h_binedges = h_bins + bin_size/2

    h_ind_bin = np.digitize(hs_i, bins=h_binedges)
    unique, counts = np.unique(h_ind_bin, return_counts=True)

    ind_min_bin = unique[counts>10][0]
    ind_max_bin = unique[counts>10][-1]
    x_bins = h_bins[ind_min_bin:ind_max_bin+1]
    real_bins = np.zeros(len(x_bins))

    weib_par_cond = np.zeros((len(x_bins),3))

    for j in range(len(x_bins)):
        mask1 = h_ind_bin == j + ind_min_bin
        real_bins[j] = hs_i[mask1].mean()
        weib_par_cond[j,:] = weibull_min.fit(v_i[mask1], floc=0)
    try:
        k = curve_fit(power3, real_bins, weib_par_cond[:,0], bounds=bounds)[0]
        a = curve_fit(power3, real_bins, weib_par_cond[:,2], bounds=bounds)[0]
    except RuntimeError: 
        try:
            k = curve_fit(power3, real_bins[0:-1], weib_par_cond[0:-1,0], bounds=bounds)[0]
            a = curve_fit(power3, real_bins[0:-1], weib_par_cond[0:-1,2], bounds=bounds)[0]
        except RuntimeError: 
            k = curve_fit(power3, real_bins[0:-2], weib_par_cond[0:-2,0], bounds=bounds)[0]
            a = curve_fit(power3, real_bins[0:-2], weib_par_cond[0:-2,2], bounds=bounds)[0]

    x1_50 = weibull_min.ppf( norm.cdf(u1_50), *weib_par2)

    # The weibull conditional distribution
    k_x1_50 =  power3(x1_50, *k)
    a_x1_50 =  power3(x1_50, *a)

    x0_50 = weibull_min.ppf( norm.cdf(u0_50), k_x1_50, loc=0, scale=a_x1_50)

#    import seaborn as sns
#    h = sns.jointplot(x= v_i , y=hs_i, s=5)
#    h.x, h.y = x0_50, x1_50
#    h.plot_joint(plt.plot, color='C1')
    
#    ids_contour_i = []
    # Compute 50-yr IFORM contour.
#    iform_contour_i = IFormContour(my_fit.mul_var_dist, return_period, 1,
#                                   NR_OF_POINTS_ON_CONTOUR)
    
    if DO_COMPUTE_CONFIDENCE_INTERVAL:
        # Define angles based on normalization.
        theta_stars = np.arange(0, 360, ANGLE_STEP_FOR_CI) / 180 * np.pi
        t1 = max(dataset_d_v) - min(dataset_d_v)
        t2 = max(dataset_d_hs) - min(dataset_d_hs)
        #print('t1: ' + str(t1))
        #print('t2: ' + str(t2))
        thetas = thetastar_to_theta(theta_stars, t1, t2)
        #print('Thetas: ' + str(thetas / np.pi * 180))
        #print('Theta_stars: ' + str(theta_stars/np.pi * 180))
        nr_of_datapoints_on_angled_line = 10
        line_tot_length = 50.0
        line_length = np.linspace(0.0, line_tot_length, nr_of_datapoints_on_angled_line)

        # Compute lines that have an angle theta to the x-axis.
        theta_line_v = list()
        theta_line_hs = list()
        theta_v = list()
        theta_hs = list()
        for j, theta in enumerate(thetas):
            theta_line_v.append(np.multiply(np.cos(theta),  line_length) + v0)
            theta_line_hs.append(np.multiply(np.sin(theta), line_length) + hs0)
#            c_v = np.append(x1_50, x1_50[0])
#            c_hs = np.append(x0_50, x0_50[0])
            theta_v_j, theta_hs_j = contour_intersection(
                theta_line_v[j], theta_line_hs[j], x0_50, x1_50, True)
            theta_v_ij[i,j] = theta_v_j
            theta_hs_ij[i,j] = theta_hs_j
#            theta_v.append(theta_v_j)
#            theta_hs.append(theta_hs_j)

    if i == 0:
        contour0, contour1 = [x0_50], [x1_50]
    else:
        contour0.append(x0_50)
        contour1.append(x1_50)
        
# Plot the environmental contours.
fig = plt.figure(figsize=(5, 5), dpi=150)
ax = fig.add_subplot(111)
for i in range(len(contour0)):
    if i == 0:
        plotted_sample = PlottedSample(x=np.asarray(dataset_d_v),
                                       y=np.asarray(dataset_d_hs),
                                       ax=ax,
                                       label='dataset D')
        contour_label = str(return_period) + '-yr contour'
        plot_contour(x=contour0[i],
                     y=contour1[i],
                     ax=ax,
                     contour_label=contour_label,
                     x_label=label_v,
                     y_label=label_hs,
                     line_style='b-',
                     alpha=0.4,
                     plotted_sample=plotted_sample)
    else:
        plot_contour(x=contour0[i],
                     y=contour1[i],
                     line_style='b-',
                     alpha=0.4,
                     ax=ax)
    if DO_COMPUTE_CONFIDENCE_INTERVAL and DO_PLOT_ANGLE_LINES:
        for j, (line_v, line_hs) in enumerate(zip(theta_line_v, theta_line_hs)):
            if i == 0:
                plt.plot(line_v, line_hs, 'r-')
            plt.plot(theta_v, theta_hs, 'gx')
if NR_OF_YEARS_TO_DRAW == 1:
    plt.title('Samples cover ' + str(NR_OF_YEARS_TO_DRAW) + ' year')
else:
    plt.title('Samples cover ' + str(NR_OF_YEARS_TO_DRAW) + ' years')
plt.xlim((0, 32))
plt.ylim((0, 14))
plt.show()
plt.savefig('../results/figures/hannesdottir_asta_exercise2_'+ str(NR_OF_YEARS_TO_DRAW) +'yr_allcontours.png', dpi=300)

if DO_COMPUTE_CONFIDENCE_INTERVAL:
#    theta_v_ij = np.zeros(shape=(len(contour0), thetas.size))
#    theta_hs_ij = np.zeros(shape=(len(contour0), thetas.size))
    distance_to_origin_ij = np.zeros(shape=(len(contour0), thetas.size))
#    for i, contour in enumerate(contours):
    for i in range(len(contour0)):
        for j in range(len(thetas)):
#        for j, (v_j, hs_j) in enumerate(zip(contour.theta_v, contour.theta_hs)):
            v_j = theta_v_ij[i, j]
            hs_j = theta_hs_ij[i, j]
            o = np.array([v0, hs0])
            p = np.array([v_j, hs_j]).flatten()
            op = p - o
            distance_to_origin_ij[i, j] = np.sqrt(op[0]*op[0] + op[1]*op[1])
    sorted_v = np.zeros(shape=(len(contour0), thetas.size))
    sorted_hs = np.zeros(shape=(len(contour0), thetas.size))
    for j in range(thetas.size):
        sorted_indices = np.argsort(distance_to_origin_ij[:, j])
        sorted_v[:, j] = theta_v_ij[sorted_indices, j]
        sorted_hs[:, j] = theta_hs_ij[sorted_indices, j]
    percentile50_index = int(round((NR_OF_BOOTSTRAP_SAMPLES - 1) * (50.0 / 100.0)))
    bottom_percentile_index = int(round((NR_OF_BOOTSTRAP_SAMPLES - 1) * (BOTTOM_PERCENTILE / 100.0)))
    upper_percentile_index = int(round((NR_OF_BOOTSTRAP_SAMPLES - 1) * (UPPER_PERCENTILE / 100.0)))

    # Save the median, bottom and upper percentile contours.
    folder_name = 'contour_coordinates/'
    file_name_median = determine_file_name_e2(
        'Asta', 'Hannesdottir', NR_OF_YEARS_TO_DRAW, 'median')
    write_contour(sorted_v[percentile50_index, :],
                  sorted_hs[percentile50_index, :],
                  folder_name + file_name_median,
                  label_x=label_v,
                  label_y=label_hs)
    file_name_bottom = determine_file_name_e2(
        'Asta', 'Hannesdottir', NR_OF_YEARS_TO_DRAW, 'bottom')
    write_contour(sorted_v[bottom_percentile_index, :],
                  sorted_hs[bottom_percentile_index, :],
                  folder_name + file_name_bottom,
                  label_x=label_v,
                  label_y=label_hs)
    file_name_upper = determine_file_name_e2(
        'Asta', 'Hannesdottir', NR_OF_YEARS_TO_DRAW, 'upper')
    write_contour(sorted_v[upper_percentile_index, :],
                  sorted_hs[upper_percentile_index, :],
                  folder_name + file_name_upper,
                  label_x=label_v,
                  label_y=label_hs)

    # Read the contours from the csv files.
    (contour_v_median, contour_hs_median) = read_contour(folder_name + file_name_median)
    (contour_v_bottom, contour_hs_bottom) = read_contour(folder_name + file_name_bottom)
    (contour_v_upper, contour_hs_upper) = read_contour(folder_name + file_name_upper)

    # Plot the sample, the median contour and the confidence interval.
    fig = plt.figure(figsize=(5, 5), dpi=150)
    ax = fig.add_subplot(111)
    plotted_sample = PlottedSample(x=np.asarray(dataset_d_v),
                                   y=np.asarray(dataset_d_hs),
                                   ax=ax,
                                   label='dataset D')
    contour_labels = ['50th percentile contour', '2.5th percentile contour',
                      '97.5th percentile contour']
    plot_confidence_interval(
        x_median=contour_v_median, y_median=contour_hs_median,
        x_bottom=contour_v_bottom, y_bottom=contour_hs_bottom,
        x_upper=contour_v_upper, y_upper=contour_hs_upper, ax=ax,
        x_label=label_v,
        y_label=label_hs, contour_labels=contour_labels,
        plotted_sample=plotted_sample)
    if NR_OF_YEARS_TO_DRAW == 1:
        plt.title('Samples cover ' + str(NR_OF_YEARS_TO_DRAW) + ' year')
    else:
        plt.title('Samples cover ' + str(NR_OF_YEARS_TO_DRAW) + ' years')
    plt.show()
    plt.savefig('../results/figures/hannesdottir_asta_exercise2_'+ str(NR_OF_YEARS_TO_DRAW) +'yr_confidenceintervals.png', dpi=300)
