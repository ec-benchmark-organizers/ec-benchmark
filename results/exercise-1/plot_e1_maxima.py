import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from palettable.colorbrewer.qualitative import Paired_11 as mycorder
from palettable.colorbrewer.qualitative import Paired_12 as mycorder12
from viroconcom.read_write import read_contour, read_ecbenchmark_dataset

from settings import lastname_firstname, legends_for_contribution, ls_for_contribution

legends_for_contribution = [l.replace('Contribution', 'Contr.') for l in legends_for_contribution]
dataset_chars = ['A', 'B', 'C', 'D', 'E', 'F']
n_contributions = len(legends_for_contribution)

colors_for_contribution = np.array(mycorder.mpl_colors) # Must be np.array for filtering.

# Marginal exceedance = 'o', total exceedance = 'v'
marker_class0 = 'o'
marker_class1 = 'v'
marker_class = np.array([marker_class0, marker_class1])
classes = np.array([1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
if n_contributions == 12:
    classes = np.append(classes, 0)
    colors_for_contribution = np.array(mycorder12.mpl_colors)

a_max_hs_c1 = np.empty(n_contributions)
a_min_tz_c1 = np.empty(n_contributions)
a_max_tz_c1 = np.empty(n_contributions)
b_max_hs_c1 =  np.empty(n_contributions)
b_min_tz_c1 = np.empty(n_contributions)
b_max_tz_c1 = np.empty(n_contributions)
c_max_hs_c1 =  np.empty(n_contributions)
c_min_tz_c1 = np.empty(n_contributions)
c_max_tz_c1 = np.empty(n_contributions)
a_max_hs_c20 =  np.empty(n_contributions)
a_min_tz_c20 = np.empty(n_contributions)
a_max_tz_c20 = np.empty(n_contributions)
b_max_hs_c20 =  np.empty(n_contributions)
b_min_tz_c20 = np.empty(n_contributions)
b_max_tz_c20 = np.empty(n_contributions)
c_max_hs_c20 =  np.empty(n_contributions)
c_min_tz_c20 = np.empty(n_contributions)
c_max_tz_c20 = np.empty(n_contributions)

d_max_v_c1 = np.empty(n_contributions)
d_max_hs_c1 = np.empty(n_contributions)
e_max_v_c1 = np.empty(n_contributions)
e_max_hs_c1 = np.empty(n_contributions)
f_max_v_c1 = np.empty(n_contributions)
f_max_hs_c1 = np.empty(n_contributions)
d_max_v_c50 = np.empty(n_contributions)
d_max_hs_c50 = np.empty(n_contributions)
e_max_v_c50 = np.empty(n_contributions)
e_max_hs_c50 = np.empty(n_contributions)
f_max_v_c50 = np.empty(n_contributions)
f_max_hs_c50 = np.empty(n_contributions)
for i in range(n_contributions):
    contribution_id = i + 1
    participant_nr = contribution_id
    if 11 >= contribution_id >= 9:
        participant_nr = 9
    elif contribution_id > 11:
        # Because contribution 9 holds 3 sets of contours.
        participant_nr = contribution_id - 2
    for dataset_char in dataset_chars:
        if dataset_char in ('A', 'B', 'C'):
            folder_name = 'results/exercise-1/contribution-' + str(participant_nr)

            # Read the 1-year contours
            return_period = 1
            file_name = folder_name + '/' + \
                        lastname_firstname[contribution_id - 1] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            (hs, tz) = read_contour(file_name)
            if participant_nr in (8, 9):
                (tz, hs) = read_contour(file_name)
            if dataset_char == 'A':
                a_max_hs_c1[i] = max(hs)
                a_min_tz_c1[i] = min(tz)
                a_max_tz_c1[i] = max(tz)
            elif dataset_char == 'B':
                b_max_hs_c1[i] = max(hs)
                b_min_tz_c1[i] = min(tz)
                b_max_tz_c1[i] = max(tz)
            elif dataset_char == 'C':
                c_max_hs_c1[i] = max(hs)
                c_min_tz_c1[i] = min(tz)
                c_max_tz_c1[i] = max(tz)

            # Now read the 20-year contours.
            return_period = 20
            file_name = folder_name + '/' + \
                        lastname_firstname[contribution_id - 1] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            (hs, tz) = read_contour(file_name)
            if participant_nr in (8, 9):
                (tz, hs) = read_contour(file_name)
            if dataset_char == 'A':
                a_max_hs_c20[i] = max(hs)
                a_min_tz_c20[i] = min(tz)
                a_max_tz_c20[i] = max(tz)
            elif dataset_char == 'B':
                b_max_hs_c20[i] = max(hs)
                b_min_tz_c20[i] = min(tz)
                b_max_tz_c20[i] = max(tz)
            elif dataset_char == 'C':
                c_max_hs_c20[i] = max(hs)
                c_min_tz_c20[i] = min(tz)
                c_max_tz_c20[i] = max(tz)
        else:
            folder_name = 'results/exercise-1/contribution-' + str(participant_nr)

            # Read the 1-yr contur.
            return_period = 1
            file_name = folder_name + '/' + \
                        lastname_firstname[contribution_id - 1] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            if participant_nr in (1, 2, 3, 5, 6, 8):
                (hs, v) = read_contour(file_name)
            else:
                (v, hs) = read_contour(file_name)
            if dataset_char == 'D':
                d_max_v_c1[i] = max(v)
                d_max_hs_c1[i] = max(hs)
            elif dataset_char == 'E':
                e_max_v_c1[i] = max(v)
                e_max_hs_c1[i] = max(hs)
            elif dataset_char == 'F':
                f_max_v_c1[i] = max(v)
                f_max_hs_c1[i] = max(hs)

            # Now read the 50-yr contur.
            return_period = 50
            file_name = folder_name + '/' + \
                        lastname_firstname[contribution_id - 1] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            if participant_nr in (1, 2, 3, 5, 6, 8):
                (hs, v) = read_contour(file_name)
            else:
                (v, hs) = read_contour(file_name)
            if dataset_char == 'D':
                d_max_v_c50[i] = max(v)
                d_max_hs_c50[i] = max(hs)
            elif dataset_char == 'E':
                e_max_v_c50[i] = max(v)
                e_max_hs_c50[i] = max(hs)
            elif dataset_char == 'F':
                f_max_v_c50[i] = max(v)
                f_max_hs_c50[i] = max(hs)

# Load the environmental data and compute their minima and maxima.
empirical_max_hs_abc = np.empty([3, 1])
empirical_min_tz_abc = np.empty([3, 1])
empirical_max_tz_abc = np.empty([3, 1])
empirical_hs1_abc = np.empty([3, 1])
empirical_tz1_abc = np.empty([3, 1])
for i, dataset_char in np.ndenumerate(['A', 'B', 'C']):
    file_name_provided = 'datasets/' + dataset_char + '.txt'
    file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'
    hs_p, tz_p, lhs, ltz = read_ecbenchmark_dataset(
        file_name_provided)
    hs_r, tz_r, lhs, ltz = read_ecbenchmark_dataset(
        file_name_retained)
    hs = np.append(hs_p, hs_r)
    tz = np.append(tz_p, tz_r)
    empirical_max_hs_abc[i] = max(hs)
    empirical_min_tz_abc[i] = min(tz)
    empirical_max_tz_abc[i] = max(tz)
    pe_1yr = 1.0 / (365.25 * 24)
    empirical_hs1_abc[i] = np.quantile(hs, 1 - pe_1yr)
    empirical_tz1_abc[i] = np.quantile(tz, 1 - pe_1yr)
empirical_max_v_def = np.empty([3, 1])
empirical_max_hs_def = np.empty([3, 1])
empirical_v1_def = np.empty([3, 1])
empirical_hs1_def = np.empty([3, 1])
for i, dataset_char in np.ndenumerate(['D', 'E', 'F']):
    file_name_provided = 'datasets/' + dataset_char + '.txt'
    file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'
    v_p, hs_p, lv, lhs = read_ecbenchmark_dataset(file_name_provided)
    v_r, hs_r, lv, lhs = read_ecbenchmark_dataset(file_name_retained)
    v = np.append(v_p, v_r)
    hs = np.append(hs_p, hs_r)
    empirical_max_v_def[i] = max(v)
    empirical_max_hs_def[i] = max(hs)
    pe_1yr = 1.0 / (365.25 * 24)
    empirical_v1_def[i] = np.quantile(v, 1 - pe_1yr)
    empirical_hs1_def[i] = np.quantile(hs, 1 - pe_1yr)

# Plot the figure for datasets A, B, C.
fig_abc, axs = plt.subplots(2, 2, figsize=(10*0.75, 9*0.75))
values = range(len(a_max_hs_c20)) + np.ones(len(a_max_hs_c20))
marker_size = 60

# Plot the 1-yr contour maxima.
for i in (0, 1):
    axs[0, 0].scatter(np.ones(np.shape(a_max_hs_c1[classes==i])), a_max_hs_c1[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i], s=marker_size,
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 0].scatter(np.ones(np.shape(b_max_hs_c1[classes==i])) + 1, b_max_hs_c1[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i], s=marker_size,
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 0].scatter(np.ones(np.shape(c_max_hs_c1[classes==i])) + 2, c_max_hs_c1[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i], s=marker_size,
                      edgecolors='k', alpha=0.7, linewidths=0.5)
for i in range(3):
    axs[0, 0].plot([i + 0.8, i + 1.2], [empirical_hs1_abc[i], empirical_hs1_abc[i]], '--k', dashes=(2, 1))
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].yaxis.set_ticks_position('left')
axs[0, 0].xaxis.set_ticks_position('bottom')
axs[0, 0].set_xticks([1, 2, 3])
axs[0, 0].set_xticklabels(['A', 'B', 'C'])
axs[0, 0].set_ylim(4, 15)
axs[0, 0].set_ylabel('Max. Hs along 1-yr contour (m)')

for i in (0, 1):
    axs[0, 1].scatter(np.ones(np.shape(a_min_tz_c1[classes==i])), a_min_tz_c1[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 1].scatter(np.ones(np.shape(b_min_tz_c1[classes==i])) + 1, b_min_tz_c1[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 1].scatter(np.ones(np.shape(c_min_tz_c1[classes==i])) + 2, c_min_tz_c1[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 1].scatter(np.ones(np.shape(a_min_tz_c1[classes==i])) + 0.2, a_max_tz_c1[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 1].scatter(np.ones(np.shape(b_min_tz_c1[classes==i])) + 1.2, b_max_tz_c1[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 1].scatter(np.ones(np.shape(c_min_tz_c1[classes==i])) + 2.2, c_max_tz_c1[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
for i in range(3):
    axs[0, 1].plot([i + 1.0, i + 1.4], [empirical_tz1_abc[i], empirical_tz1_abc[i]], '--k', dashes=(2, 1))

# Remove axis on the right and on the top (Matlab 'box off').
axs[0, 1].spines['right'].set_visible(False)
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].yaxis.set_ticks_position('left')
axs[0, 1].xaxis.set_ticks_position('bottom')

axs[0, 1].set_xticks([1.1, 2.1, 3.1])
axs[0, 1].set_xticklabels(['A', 'B', 'C'])
axs[0, 1].set_ylim(0, 20)
axs[0, 1].set_yticks([0, 5, 10, 15, 20])
axs[0, 1].set_ylabel('Min. and max Tz  along 1-yr contour (s)')

# Plot the 20-yr contour maxima.
for i in (0, 1):
    scatterHs = axs[1, 0].scatter(np.ones(np.shape(a_max_hs_c20[classes == i])),
                                  a_max_hs_c20[classes == i],
                                  marker=marker_class[i],
                                  c=colors_for_contribution[classes == i],
                                  s=marker_size,
                                  edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 0].scatter(np.ones(np.shape(b_max_hs_c20[classes==i])) + 1, b_max_hs_c20[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i], s=marker_size,
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 0].scatter(np.ones(np.shape(c_max_hs_c20[classes==i])) + 2, c_max_hs_c20[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i], s=marker_size,
                      edgecolors='k', alpha=0.7, linewidths=0.5)
i = 0
emp = axs[1, 0].plot([i + 0.8, i + 1.2], [empirical_max_hs_abc[i], empirical_max_hs_abc[i]], '-k')
for i in (1, 2):
    axs[1, 0].plot([i + 0.8, i + 1.2], [empirical_max_hs_abc[i], empirical_max_hs_abc[i]], '-k')
axs[1, 0].spines['right'].set_visible(False)
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].yaxis.set_ticks_position('left')
axs[1, 0].xaxis.set_ticks_position('bottom')
axs[1, 0].set_xticks([1, 2, 3])
axs[1, 0].set_xticklabels(['A', 'B', 'C'])
axs[1, 0].set_ylim(4, 15)
axs[1, 0].set_ylabel('Max. Hs along 20-yr contour (m)')

for i in (0, 1):
    axs[1, 1].scatter(np.ones(np.shape(a_min_tz_c20[classes == i])), a_min_tz_c20[classes == i],
                      marker=marker_class[i], c=colors_for_contribution[classes == i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 1].scatter(np.ones(np.shape(b_min_tz_c20[classes==i])) + 1, b_min_tz_c20[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 1].scatter(np.ones(np.shape(c_min_tz_c20[classes==i])) + 2, c_min_tz_c20[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 1].scatter(np.ones(np.shape(a_min_tz_c20[classes==i])) + 0.2, a_max_tz_c20[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 1].scatter(np.ones(np.shape(b_min_tz_c20[classes==i])) + 1.2, b_max_tz_c20[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 1].scatter(np.ones(np.shape(c_min_tz_c20[classes==i])) + 2.2, c_max_tz_c20[classes==i],
                      marker=marker_class[i], c=colors_for_contribution[classes==i],
                      s=marker_size, edgecolors='k', alpha=0.7, linewidths=0.5)
for i in range(3):
    axs[1, 1].plot([i + 0.8, i + 1.2], [empirical_min_tz_abc[i], empirical_min_tz_abc[i]], '-k')
    axs[1, 1].plot([i + 1.0, i + 1.4], [empirical_max_tz_abc[i], empirical_max_tz_abc[i]], '-k')

# Remove axis on the right and on the top (Matlab 'box off').
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].yaxis.set_ticks_position('left')
axs[1, 1].xaxis.set_ticks_position('bottom')

axs[1, 1].set_xticks([1.1, 2.1, 3.1])
axs[1, 1].set_xticklabels(['A', 'B', 'C'])
axs[1, 1].set_ylim(0, 20)
axs[1, 1].set_yticks([0, 5, 10, 15, 20])
axs[1, 1].set_ylabel('Min. and max Tz  along 20-yr contour (s)')

handles = []
for i in range(n_contributions):
    handles.append(
        Line2D([0], [0], marker=marker_class[classes[i]], c='w', ms=7,
               markerfacecolor=colors_for_contribution[i],
               markeredgecolor='k', alpha=0.7, linewidth=0.3))
handles.append(Line2D([0], [0], c='k', ls='--', dashes=(2,1)))
handles.append(Line2D([0], [0], c='k'))

labels = legends_for_contribution.copy()
labels.append('Empirical marginal 1-yr return value')
labels.append('Max/Min in full dataset')
fig_abc.legend(handles=handles, labels=labels,
           prop={'size': 6}, loc='lower center', ncol=7, scatterpoints=1)
fig_abc.tight_layout(rect=(0, 0.05, 1, 1))

# Plot the figure for datasets D, E F.
fig_def, axs = plt.subplots(2, 2, figsize=(10*0.75, 9*0.75))

# First, plot the 1-yr maxima.
for i in (0, 1):
    axs[0, 0].scatter(np.ones(np.shape(d_max_v_c1[classes==i])), d_max_v_c1[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 0].scatter(np.ones(np.shape(e_max_v_c1[classes==i])) + 1, e_max_v_c1[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 0].scatter(np.ones(np.shape(f_max_v_c1[classes==i])) + 2, f_max_v_c1[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)

for i in  range(3):
    axs[0, 0].plot([i + 0.8, i + 1.2], [empirical_v1_def[i], empirical_v1_def[i]], '-k', dashes=(2,1))
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].yaxis.set_ticks_position('left')
axs[0, 0].xaxis.set_ticks_position('bottom')
axs[0, 0].set_xticks([1, 2, 3])
axs[0, 0].set_xticklabels(['D', 'E', 'F'])
axs[0, 0].set_ylim(21, 34)
axs[0, 0].set_ylabel('Max. V  along 1-yr contour (m/s)')

for i in (0, 1):
    axs[0, 1].scatter(np.ones(np.shape(d_max_hs_c1[classes==i])), d_max_hs_c1[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 1].scatter(np.ones(np.shape(e_max_hs_c1[classes==i])) + 1, e_max_hs_c1[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[0, 1].scatter(np.ones(np.shape(f_max_hs_c1[classes==i])) + 2, f_max_hs_c1[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
for i in range(3):
    axs[0, 1].plot([i + 0.8, i + 1.2], [empirical_hs1_def[i], empirical_hs1_def[i]], '-k', dashes=(2,1))
axs[0, 1].spines['right'].set_visible(False)
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].yaxis.set_ticks_position('left')
axs[0, 1].xaxis.set_ticks_position('bottom')
axs[0, 1].set_xticks([1, 2, 3])
axs[0, 1].set_xticklabels(['D', 'E', 'F'])
axs[0, 1].set_ylim(6, 20)
axs[0, 1].set_ylabel('Max. Hs  along 1-yr contour  (m)')

# Now plot the 50-yr maxima.
for i in (0, 1):
    axs[1, 0].scatter(np.ones(np.shape(d_max_v_c50[classes==i])), d_max_v_c50[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 0].scatter(np.ones(np.shape(e_max_v_c50[classes==i])) + 1, e_max_v_c50[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 0].scatter(np.ones(np.shape(f_max_v_c50[classes==i])) + 2, f_max_v_c50[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)

for i in range(3):
    axs[1, 0].plot([i + 0.8, i + 1.2], [empirical_max_v_def[i], empirical_max_v_def[i]], '-k')
axs[1, 0].spines['right'].set_visible(False)
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].yaxis.set_ticks_position('left')
axs[1, 0].xaxis.set_ticks_position('bottom')
axs[1, 0].set_xticks([1, 2, 3])
axs[1, 0].set_xticklabels(['D', 'E', 'F'])
axs[1, 0].set_ylim(21, 34)
axs[1, 0].set_ylabel('Max. V  along 50-yr contour (m/s)')

for i in (0, 1):
    axs[1, 1].scatter(np.ones(np.shape(d_max_hs_c50[classes==i])), d_max_hs_c50[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 1].scatter(np.ones(np.shape(e_max_hs_c50[classes==i])) + 1, e_max_hs_c50[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
    axs[1, 1].scatter(np.ones(np.shape(f_max_hs_c50[classes==i])) + 2, f_max_hs_c50[classes==i],
                      c=colors_for_contribution[classes==i], s=marker_size, marker=marker_class[i],
                      edgecolors='k', alpha=0.7, linewidths=0.5)
for i in range(3):
    axs[1, 1].plot([i + 0.8, i + 1.2], [empirical_max_hs_def[i], empirical_max_hs_def[i]], '-k')
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].yaxis.set_ticks_position('left')
axs[1, 1].xaxis.set_ticks_position('bottom')
axs[1, 1].set_xticks([1, 2, 3])
axs[1, 1].set_xticklabels(['D', 'E', 'F'])
axs[1, 1].set_ylim(6, 20)
axs[1, 1].set_ylabel('Max. Hs  along 50-yr contour  (m)')

labels = legends_for_contribution.copy()
labels.append('Empirical marginal 1-yr return value')
labels.append('Maximum in full dataset')
fig_def.legend(handles=handles, labels=labels,
           prop={'size': 6}, loc='lower center', ncol=7, scatterpoints=1)
fig_def.tight_layout(rect=(0, 0.05, 1, 1))

plt.show()
fig_abc.savefig('results/exercise-1/e1_max_values_abc.pdf')
fig_def.savefig('results/exercise-1/e1_max_values_def.pdf')
