import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from palettable.colorbrewer.qualitative import Paired_11 as mycorder
from viroconcom.read_write import read_contour, read_ecbenchmark_dataset

dataset_chars = ['A', 'B', 'C', 'D', 'E', 'F']
lastname_firstname = ['Wei_Bernt', 'GC_CGS', 'hannesdottir_asta',
                      'haselsteiner_andreas', 'BV', 'mackay_ed',
                      'qiao_chi', 'rode_anna', 'vanem_DirectSampling',
                      'vanem_DirectSamplingsmoothed', 'vanem_IFORM']
legends_for_contribution = ['Contribution 1',
                          'Contribution 2',
                          'Contribution 3',
                          'Contribution 4',
                          'Contribution 5',
                          'Contribution 6',
                          'Contribution 7',
                          'Contribution 8',
                          'Contribution 9, DS',
                          'Contribution 9, DS smoothed',
                          'Contribution 9, IFORM'
                          ]

a_max_hs =  []
a_min_tz = []
a_max_tz = []
b_max_hs =  []
b_min_tz = []
b_max_tz = []
c_max_hs =  []
c_min_tz = []
c_max_tz = []
d_max_v = []
d_max_hs = []
e_max_v = []
e_max_hs = []
f_max_v = []
f_max_hs = []
for i in range(11):
    contribution_id = i + 1
    if contribution_id > 9:
        participant_nr = 9
    else:
        participant_nr = contribution_id
    for dataset_char in dataset_chars:
        if dataset_char in ('A', 'B', 'C'):
            return_period = 20
            folder_name = 'results/exercise-1/contribution-' + str(participant_nr)
            file_name = folder_name + '/' + \
                        lastname_firstname[contribution_id - 1] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            (hs, tz) = read_contour(file_name)
            if participant_nr in (8, 9):
                (tz, hs) = read_contour(file_name)
            if dataset_char == 'A':
                a_max_hs.append(max(hs))
                a_min_tz.append(min(tz))
                a_max_tz.append(max(tz))
            elif dataset_char == 'B':
                b_max_hs.append(max(hs))
                b_min_tz.append(min(tz))
                b_max_tz.append(max(tz))
            elif dataset_char == 'C':
                c_max_hs.append(max(hs))
                c_min_tz.append(min(tz))
                c_max_tz.append(max(tz))
        else:
            return_period = 50
            folder_name = 'results/exercise-1/contribution-' + str(participant_nr)
            file_name = folder_name + '/' + \
                        lastname_firstname[contribution_id - 1] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            if participant_nr in (1, 2, 3, 5, 6, 8):
                (hs, v) = read_contour(file_name)
            else:
                (v, hs) = read_contour(file_name)
            if dataset_char == 'D':
                d_max_v.append(max(v))
                d_max_hs.append(max(hs))
            elif dataset_char == 'E':
                e_max_v.append(max(v))
                e_max_hs.append(max(hs))
            elif dataset_char == 'F':
                f_max_v.append(max(v))
                f_max_hs.append(max(hs))

# Load the environmental data and compute their minima and maxima.
empirical_max_hs_abc = np.empty([3, 1])
empirical_min_tz_abc = np.empty([3, 1])
empirical_max_tz_abc = np.empty([3, 1])
for i, dataset_char in np.ndenumerate(['A', 'B', 'C']):
    file_name_provided = 'datasets/' + dataset_char + '.txt'
    file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'
    hs_p, tz_p, lhs, ltz = read_ecbenchmark_dataset(
        file_name_provided)
    hs_r, tz_r, lhs, ltz = read_ecbenchmark_dataset(
        file_name_retained)
    empirical_max_hs_abc[i] = max([max(hs_p), max(hs_r)])
    empirical_min_tz_abc[i] = max([min(tz_p), min(tz_r)])
    empirical_max_tz_abc[i] = max([max(tz_p), max(tz_r)])
empirical_max_v_def = np.empty([3, 1])
empirical_max_hs_def = np.empty([3, 1])
for i, dataset_char in np.ndenumerate(['D', 'E', 'F']):
    file_name_provided = 'datasets/' + dataset_char + '.txt'
    file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'
    v_p, hs_p, lv, lhs = read_ecbenchmark_dataset(file_name_provided)
    v_r, hs_r, lv, lhs = read_ecbenchmark_dataset(file_name_retained)
    empirical_max_v_def[i] = max([max(v_p), max(v_r)])
    empirical_max_hs_def[i] = max([max(hs_p), max(hs_r)])

# Plot the figure for datasets A, B, C.
fig_abc, axs = plt.subplots(1, 2, figsize=(10*0.75, 5*0.75))
values = range(len(a_max_hs)) + np.ones(len(a_max_hs))
marker_size = 60

scatterHs = axs[0].scatter(np.ones(np.shape(a_max_hs)), a_max_hs, c=values, s=marker_size,
                              cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[0].scatter(np.ones(np.shape(b_max_hs)) + 1, b_max_hs, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[0].scatter(np.ones(np.shape(c_max_hs)) + 2, c_max_hs, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
for i in range(3):
    axs[0].plot([i + 0.8, i + 1.2], [empirical_max_hs_abc[i], empirical_max_hs_abc[i]], '-k')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].yaxis.set_ticks_position('left')
axs[0].xaxis.set_ticks_position('bottom')
axs[0].set_xticks([1, 2, 3])
axs[0].set_xticklabels(['A', 'B', 'C'])
axs[0].set_ylabel('Max. Hs along 20-yr contour (m)')


scatterTp = axs[1].scatter(np.ones(np.shape(a_min_tz)), a_min_tz, c=values, s=marker_size,
                              cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[1].scatter(np.ones(np.shape(b_min_tz)) + 1, b_min_tz, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[1].scatter(np.ones(np.shape(c_min_tz)) + 2, c_min_tz, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[1].scatter(np.ones(np.shape(a_min_tz)) + 0.2, a_max_tz, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[1].scatter(np.ones(np.shape(b_min_tz)) + 1.2, b_max_tz, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[1].scatter(np.ones(np.shape(c_min_tz)) + 2.2, c_max_tz, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
for i in range(3):
    axs[1].plot([i + 0.9, i + 1.1], [empirical_min_tz_abc[i], empirical_min_tz_abc[i]], '-k')
    axs[1].plot([i + 1.1, i + 1.3], [empirical_max_tz_abc[i], empirical_max_tz_abc[i]], '-k')
    # Remove axis on the right and on the top (Matlab 'box off').
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].yaxis.set_ticks_position('left')
axs[1].xaxis.set_ticks_position('bottom')
axs[1].set_xticks([1.1, 2.1, 3.1])
axs[1].set_xticklabels(['A', 'B', 'C'])
axs[1].set_ylabel('Min. and max Tz  along 20-yr contour (s)')

fig_abc.legend(handles=scatterHs.legend_elements()[0], labels=legends_for_contribution,
           prop={'size': 6}, loc='lower center', ncol=6, scatterpoints=1)
fig_abc.tight_layout(rect=(0, 0.05, 1, 1))

# Plot the figure for datasets D, E F.
fig_def, axs = plt.subplots(1, 2, figsize=(10*0.75, 5*0.75))
scatterV = axs[0].scatter(np.ones(np.shape(d_max_v)), d_max_v, c=values, s=marker_size,
                              cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[0].scatter(np.ones(np.shape(e_max_v)) + 1, e_max_v, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[0].scatter(np.ones(np.shape(f_max_v)) + 2, f_max_v, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
for i in range(3):
    axs[0].plot([i + 0.8, i + 1.2], [empirical_max_v_def[i], empirical_max_v_def[i]], '-k')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].yaxis.set_ticks_position('left')
axs[0].xaxis.set_ticks_position('bottom')
axs[0].set_xticks([1, 2, 3])
axs[0].set_xticklabels(['D', 'E', 'F'])
axs[0].set_ylabel('Max. V  along 50-yr contour (m/s)')


scatterHs = axs[1].scatter(np.ones(np.shape(d_max_hs)), d_max_hs, c=values, s=marker_size,
                              cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[1].scatter(np.ones(np.shape(e_max_hs)) + 1, e_max_hs, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
axs[1].scatter(np.ones(np.shape(f_max_hs)) + 2, f_max_hs, c=values, s=marker_size,
                  cmap=mycorder.mpl_colormap, edgecolors='k', alpha=0.7, linewidths=0.5)
for i in range(3):
    axs[1].plot([i + 0.8, i + 1.2], [empirical_max_hs_def[i], empirical_max_hs_def[i]], '-k')
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].yaxis.set_ticks_position('left')
axs[1].xaxis.set_ticks_position('bottom')
axs[1].set_xticks([1, 2, 3])
axs[1].set_xticklabels(['D', 'E', 'F'])
axs[1].set_ylabel('Max. Hs  along 50-yr contour  (m)')

fig_def.legend(handles=scatterHs.legend_elements()[0], labels=legends_for_contribution,
           prop={'size': 6}, loc='lower center', ncol=6, scatterpoints=1)
fig_def.tight_layout(rect=(0, 0.05, 1, 1))

plt.show()
fig_abc.savefig('results/e1_max_values_abc.pdf')
fig_def.savefig('results/e1_max_values_def.pdf')
