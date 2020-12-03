import matplotlib.pyplot as plt
import numpy as np
from palettable.colorbrewer.qualitative import Paired_9 as mycorder
from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.plot import plot_contour

from settings import lastname_firstname, legends_for_contribution, ls_for_contribution

dataset_chars = ['A', 'B', 'C']
return_periods = [1, 20]
n_contours_to_analyze = len(legends_for_contribution)

colors_for_contribution = mycorder.mpl_colors
for idx in range(2):
        colors_for_contribution.append(colors_for_contribution[8])
colors_for_contribution.append('blue')

fig, axs = plt.subplots(len(return_periods), len(dataset_chars), sharex='row', sharey='row', figsize=(10, 8))
max_hs_of_sample = 0
for (return_period, ax0) in zip(return_periods, axs):
    for (dataset_char, ax1) in zip(dataset_chars, ax0):
        # Load the environmental data.
        file_name_provided = 'datasets/' + dataset_char + '.txt'
        file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'
        hs_p, tz_p, label_hs, label_tz = read_ecbenchmark_dataset(file_name_provided)
        hs_r, tz_r, label_hs, label_tz = read_ecbenchmark_dataset(file_name_retained)
        max_hs_of_sample = max([max_hs_of_sample, max(hs_p), max(hs_r)])

        contours_hs = []
        contours_tz = []
        max_hs_on_contour = np.empty(n_contours_to_analyze)
        for i in range(n_contours_to_analyze):
            contribution_nr = i + 1
            if 11 >= contribution_nr >= 9:
                contribution_nr = 9
            elif contribution_nr > 11:
                # Because contribution 9 holds 3 sets of contours.
                contribution_nr = contribution_nr - 2
            folder_name = 'results/exercise-1/contribution-' + str(contribution_nr)
            file_name = folder_name + '/' + lastname_firstname[i] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            (hs, tz) = read_contour(file_name)
            if i in (7, 8, 9, 10):
                (tz, hs) = read_contour(file_name)
            contours_hs.append(hs)
            contours_tz.append(tz)
            max_hs_on_contour[i] = max(hs[~np.isnan(tz)])

        # Plot the data and the contour.
        ax1.scatter(tz_p, hs_p, c='black', alpha=0.5, zorder=-2)
        ax1.scatter(tz_r, hs_r, marker='v', facecolor='None',
                    edgecolor='black', alpha=0.5, zorder=-2)
        for i in range(n_contours_to_analyze):
            ylim = 1.05 * max([max(max_hs_on_contour), max_hs_of_sample])
            plot_contour(contours_tz[i], contours_hs[i],
                         ax=ax1,
                         color=colors_for_contribution[i],
                         linestyle=ls_for_contribution[i],
                         upper_ylim=ylim)
        
        ax1.set_rasterization_zorder(-1)
        ax1.set_title('Dataset ' + dataset_char + ', ' + str(return_period) + '-yr contour')

for ax in axs[1]:
    ax.set_xlabel(label_tz.capitalize())
for ax in axs[:,0]:
    ax.set_ylabel(label_hs.capitalize())


lgd = fig.legend(legends_for_contribution, 
           loc='lower center', 
           ncol=6, 
           prop={'size': 8})
fig.tight_layout(rect=(0,0.05,1,1))
plt.savefig('results/exercise-1/e1_overlay_abc.pdf', bbox_inches='tight', bbox_extra_artists=[lgd])
plt.show()
