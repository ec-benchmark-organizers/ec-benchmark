import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from viroconcom.read_write import read_contour, read_ecbenchmark_dataset

from settings import lastname_firstname, legends_for_contribution, ls_for_contribution, contour_classes, colors_for_contribution

legends_for_contribution = [l.replace('Contribution', 'Contr.') for l in legends_for_contribution]
dataset_chars = ['A', 'B', 'C', 'D', 'E', 'F']
n_contributions = len(legends_for_contribution) - 2

# Marginal exceedance = 'o', total exceedance = 'v'
marker_class0 = 'o'
marker_class1 = 'v'
marker_classes = np.array([marker_class0, marker_class1])
class0_color = '#1c5abd' # blue
class1_color = '#bd1c1c' # red
color_classes = np.array([class0_color, class1_color]) 
classes = contour_classes

d_min_v_c1 = np.empty(n_contributions)
d_min_hs_c1 = np.empty(n_contributions)
e_min_v_c1 = np.empty(n_contributions)
e_min_hs_c1 = np.empty(n_contributions)
f_min_v_c1 = np.empty(n_contributions)
f_min_hs_c1 = np.empty(n_contributions)
d_min_v_c50 = np.empty(n_contributions)
d_min_hs_c50 = np.empty(n_contributions)
e_min_v_c50 = np.empty(n_contributions)
e_min_hs_c50 = np.empty(n_contributions)
f_min_v_c50 = np.empty(n_contributions)
f_min_hs_c50 = np.empty(n_contributions)

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
        if dataset_char in ('D', 'E', 'F'):
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
                d_min_v_c1[i] = min(v)
                d_min_hs_c1[i] = min(hs)
                d_max_v_c1[i] = max(v)
                d_max_hs_c1[i] = max(hs)
            elif dataset_char == 'E':
                e_min_v_c1[i] = min(v)
                e_min_hs_c1[i] = min(hs)
                e_max_v_c1[i] = max(v)
                e_max_hs_c1[i] = max(hs)
            elif dataset_char == 'F':
                f_min_v_c1[i] = min(v)
                f_min_hs_c1[i] = min(hs)
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
                d_min_v_c50[i] = min(v)
                d_min_hs_c50[i] = min(hs)
                d_max_v_c50[i] = max(v)
                d_max_hs_c50[i] = max(hs)
            elif dataset_char == 'E':
                e_min_v_c50[i] = min(v)
                e_min_hs_c50[i] = min(hs)
                e_max_v_c50[i] = max(v)
                e_max_hs_c50[i] = max(hs)
            elif dataset_char == 'F':
                f_min_v_c50[i] = min(v)
                f_min_hs_c50[i] = min(hs)
                f_max_v_c50[i] = max(v)
                f_max_hs_c50[i] = max(hs)

# Load the environmental data and compute their minima and maxima.
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

# Plot the figure for datasets D, E, F.
fig_1, axs_1 = plt.subplots(2, 1, figsize=(7.5, 6))
fig_20, ax_50 = plt.subplots(2, 1, figsize=(7.5, 6))

# set width of bars
barWidth = 0.05
 
def autolabel(ax, rects, label, vertical_pos):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(label),
                    xy=(rect.get_x() + rect.get_width() / 2, vertical_pos),
                    xytext=(0, 0),  # points offset
                    textcoords="offset points",
                    fontsize=6,
                    color='white',
                    ha='center', va='bottom')

r = np.empty((n_contributions, 3))
# Regroup by contribution
for i in range(n_contributions):
    hs1_min = [d_min_hs_c1[i], e_min_hs_c1[i], f_min_hs_c1[i]]
    hs1_max = [d_max_hs_c1[i], e_max_hs_c1[i], f_max_hs_c1[i]]
    v1_min = [e_min_v_c1[i], e_min_v_c1[i], f_min_v_c1[i]]
    v1_max = [e_max_v_c1[i], e_max_v_c1[i], f_max_v_c1[i]]

    hs50_min = [d_min_hs_c50[i], e_min_hs_c50[i], f_min_hs_c50[i]]
    hs50_max = [d_max_hs_c50[i], e_max_hs_c50[i], f_max_hs_c50[i]]
    v50_min = [d_min_v_c50[i], e_min_v_c50[i], f_min_v_c50[i]]
    v50_max = [d_max_v_c50[i], e_max_v_c50[i], f_max_v_c50[i]]

    if i == 0:
        r[i,:] = np.arange(3)
    else:
        r[i,:] = [x + barWidth for x in r[i-1,:]]

    rects1 = axs_1[1].bar(r[i], np.subtract(hs1_max, hs1_min), bottom=hs1_min, color=color_classes[classes[i]], width=barWidth, edgecolor='white', label='max(H_s)')
    rects2 = axs_1[0].bar(r[i], np.subtract(v1_max, v1_min), bottom=v1_min, color=color_classes[classes[i]], width=barWidth, edgecolor='white', label='max(T_z)')
    autolabel(axs_1[1], rects1, i + 1, 3)
    autolabel(axs_1[0], rects2, i + 1, 8)

    rects1 = ax_50[1].bar(r[i], np.subtract(hs50_max, hs50_min), bottom=hs50_min, color=color_classes[classes[i]], width=barWidth, edgecolor='white', label='max(H_s)')
    rects2 = ax_50[0].bar(r[i], np.subtract(v50_max, v50_min), bottom=v50_min, color=color_classes[classes[i]], width=barWidth, edgecolor='white', label='max(T_z)')
    autolabel(ax_50[1], rects1, i + 1, 3)
    autolabel(ax_50[0], rects2, i + 1, 8)

for i in range(3):
    axs_1[1].plot([i - 0.5 * barWidth, i + 8.5 * barWidth], [empirical_hs1_def[i], empirical_hs1_def[i]], '--k')
    axs_1[0].plot([i - 0.5 * barWidth, i + 8.5 * barWidth], [empirical_v1_def[i], empirical_v1_def[i]], '--k')
    ax_50[1].plot([i - 0.5 * barWidth, i + 8.5 * barWidth], [empirical_max_hs_def[i], empirical_max_hs_def[i]], '--k')
    ax_50[0].plot([i - 0.5 * barWidth, i + 8.5 * barWidth], [empirical_max_v_def[i], empirical_max_v_def[i]], '--k')

axs_1[1].set_ylabel('$H_s$ range of 1-yr contour (m)') 
axs_1[0].set_ylabel('$U_{10}$ range of 1-yr contour (m/s)') 
ax_50[1].set_ylabel('$H_s$ range of 50-yr contour (m)') 
ax_50[0].set_ylabel('$U_{10}$ range of 50-yr contour (m/s)') 

# Xticks on the middle of the group bars
for axs in [axs_1, ax_50]:
    axs[0].xaxis.set_tick_params(
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    axs[1].set_xticks([r + 4 * barWidth for r in range(3)])
    axs[1].set_xticklabels(['D', 'E', 'F'])
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    axs[1].set_xlabel('Dataset')


patch_marg = Patch(facecolor=class0_color, edgecolor='w',
                        label='Marginal exceedance contour')
patch_tot = Patch(facecolor=class1_color, edgecolor='w',
                        label='Total exceedance contour')
line_observed_rp1 = Line2D([0], [0], color='k', lw=1, linestyle='--', 
                        label='Empirical marginal 1-yr return value')
line_observed_rp20 = Line2D([0], [0], color='k', lw=1, linestyle='--', 
                        label='Maximum in full dataset')
axs_1[0].legend(handles=[patch_marg, patch_tot, line_observed_rp1], 
    loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.25),)
ax_50[0].legend(handles=[patch_marg, patch_tot, line_observed_rp20], 
    loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.25),)

plt.show()
fig_1.savefig('results/exercise-1/gfx/e1_max_values_def_rp1.pdf',  bbox_inches = 'tight',
    pad_inches = 0)
fig_20.savefig('results/exercise-1/gfx/e1_max_values_def_rp50.pdf',  bbox_inches = 'tight',
    pad_inches = 0)
