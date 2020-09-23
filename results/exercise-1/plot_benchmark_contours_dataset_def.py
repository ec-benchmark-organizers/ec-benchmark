import matplotlib.pyplot as plt
import numpy as np
from palettable.colorbrewer.qualitative import Paired_9 as mycorder
from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.plot import plot_contour

dataset_chars = ['D', 'E', 'F']
return_periods = [1, 50]
lastname_firstname = ['Wei_Bernt', 'GC_CGS', 'hannesdottir_asta',
                      'haselsteiner_andreas', 'BV', 'mackay_ed',
                      'qiao_chi', 'rode_anna', 'vanem_DirectSampling',
                      'vanem_DirectSamplingsmoothed', 'vanem_IFORM']
ls_for_contribution = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '--', '-.']
colors_for_contribution = mycorder.mpl_colors
for idx in range(3):
        colors_for_contribution.append(colors_for_contribution[8])
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
n_contours_to_analyze = 11

fig, ax = plt.subplots(len(return_periods), len(dataset_chars), sharex='row', sharey='row', figsize=(10, 8))
max_hs_of_sample = 0
for (return_period, ax0) in zip(return_periods, ax):
    for (dataset_char, ax1) in zip(dataset_chars, ax0):
        # Load the environmental data.
        file_name_provided = 'datasets/' + dataset_char + '.txt'
        file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'
        v_p, hs_p, label_v, label_hs = read_ecbenchmark_dataset(file_name_provided)
        v_r, hs_r, label_v, label_hs = read_ecbenchmark_dataset(file_name_retained)
        max_hs_of_sample = max([max_hs_of_sample, max(hs_p), max(hs_r)])

        contours_v = []
        contours_hs = []
        max_hs_on_contours = np.empty(n_contours_to_analyze)
        for i in range(n_contours_to_analyze):
            contribution_nr = i + 1
            if contribution_nr > 9:
                contribution_nr = 9
            folder_name = 'results/exercise-1/contribution-' + str(contribution_nr)
            file_name = folder_name + '/' + lastname_firstname[i] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            if contribution_nr in (1, 2, 3, 5, 6, 8):
                (hs, v) = read_contour(file_name)
            else:
                (v, hs) = read_contour(file_name)
            contours_v.append(v)
            contours_hs.append(hs)
            max_hs_on_contours[i] = max(hs[~np.isnan(hs)])

        # Plot the data and the contour.
        ax1.scatter(v_p, hs_p, c='black', alpha=0.5, zorder=-2)
        ax1.scatter(v_r, hs_r, marker='v', facecolor='None',
                    edgecolor='black', alpha=0.5, zorder=-2)
        for i in range(n_contours_to_analyze):
            ylim = 1.05 * max([max(max_hs_on_contours), max_hs_of_sample])
            plot_contour(contours_v[i], contours_hs[i],
                         ax=ax1,
                         color=colors_for_contribution[i],
                         linestyle=ls_for_contribution[i],
                         upper_ylim=ylim)

        ax1.set_rasterization_zorder(-1)
        ax1.set_xlabel(label_v.capitalize())
        ax1.set_ylabel(label_hs.capitalize())
        ax1.set_title('Dataset ' + dataset_char + ', ' + str(return_period) + '-year contour')
        
lgd = fig.legend(legends_for_contribution, 
           loc='lower center', 
           ncol=6, 
           prop={'size': 8})
fig.tight_layout(rect=(0,0.05,1,1))
plt.savefig('results/e1_overlay_def.pdf', bbox_inches='tight', bbox_extra_artists=[lgd])
plt.show()
