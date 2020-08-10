import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from viroconcom.read_write import  read_contour

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


fig, axs = plt.subplots(2, 2, figsize=(12, 9))
values = range(len(a_max_hs)) + np.ones(len(a_max_hs))
marker_size = 60
scatterHs = axs[0, 0].scatter(np.ones(np.shape(a_max_hs)), a_max_hs, c=values, s=marker_size,
                              cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[0, 0].scatter(np.ones(np.shape(b_max_hs)) + 1, b_max_hs, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[0, 0].scatter(np.ones(np.shape(c_max_hs)) + 2, c_max_hs, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[0, 0].set_xticks([1, 2, 3])
axs[0, 0].set_xticklabels(['A', 'B', 'C'])
axs[0, 0].set_ylabel('Max. Hs (m)')
fig.legend(handles=scatterHs.legend_elements()[0], labels=legends_for_contribution,
           prop={'size': 6}, loc='lower center', ncol=6)

scatterTp = axs[0, 1].scatter(np.ones(np.shape(a_min_tz)), a_min_tz, c=values, s=marker_size,
                              cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[0, 1].scatter(np.ones(np.shape(b_min_tz)) + 1, b_min_tz, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[0, 1].scatter(np.ones(np.shape(c_min_tz)) + 2, c_min_tz, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[0, 1].scatter(np.ones(np.shape(a_min_tz)) + 0.2, a_max_tz, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[0, 1].scatter(np.ones(np.shape(b_min_tz)) + 1.2, b_max_tz, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[0, 1].scatter(np.ones(np.shape(c_min_tz)) + 2.2, c_max_tz, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[0, 1].set_xticks([1.1, 2.1, 3.1])
axs[0, 1].set_xticklabels(['A', 'B', 'C'])
axs[0, 1].set_ylabel('Min. and max Tp (m)')

scatterV = axs[1, 0].scatter(np.ones(np.shape(d_max_v)), d_max_v, c=values, s=marker_size,
                              cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[1, 0].scatter(np.ones(np.shape(e_max_v)) + 1, e_max_v, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[1, 0].scatter(np.ones(np.shape(f_max_v)) + 2, f_max_v, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[1, 0].set_xticks([1, 2, 3])
axs[1, 0].set_xticklabels(['D', 'E', 'F'])
axs[1, 0].set_ylabel('Max. V (m/s)')


scatterHs = axs[1, 1].scatter(np.ones(np.shape(d_max_hs)), d_max_hs, c=values, s=marker_size,
                              cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[1, 1].scatter(np.ones(np.shape(e_max_hs)) + 1, e_max_hs, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[1, 1].scatter(np.ones(np.shape(f_max_hs)) + 2, f_max_hs, c=values, s=marker_size,
                  cmap=cm.jet_r, edgecolors='k', alpha=0.7)
axs[1, 1].set_xticks([1, 2, 3])
axs[1, 1].set_xticklabels(['D', 'E', 'F'])
axs[1, 1].set_ylabel('Max. Hs (m)')

plt.show()
fig.savefig('e1_max_values', dpi=150)
