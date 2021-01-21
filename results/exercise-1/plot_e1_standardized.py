import matplotlib.pyplot as plt
import numpy as np

from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.plot import plot_contour, SamplePlotData
from viroconcom.contour_analysis import points_outside

from settings import lastname_firstname, legends_for_contribution, ls_for_contribution

# Set these constant to choose if the provided, the retained or both datasets
# should be plotted together with the contours.
DO_PLOT_PROVIDED = False
DO_PLOT_RETAINED = True

dataset_chars = ['A', 'B', 'C', 'D', 'E', 'F']
n_contours_to_analyze = len(legends_for_contribution)

for i in range(n_contours_to_analyze):
    contribution_id = i + 1

    print('Starting analysis for ' + legends_for_contribution[contribution_id - 1] +
          ' (' + lastname_firstname[contribution_id -1] + ')')

    fig, axs = plt.subplots(2, 3, figsize=(16, 10))

    dataset_count = 0
    for dataset_char in dataset_chars:
        file_name_provided = 'datasets/' + dataset_char + '.txt'
        file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'

        if dataset_char in ('A', 'B', 'C'):
            hs_p, tz_p, label_hs, label_tz = read_ecbenchmark_dataset(file_name_provided)
            hs_r, tz_r, label_hs, label_tz = read_ecbenchmark_dataset(file_name_retained)

            if DO_PLOT_PROVIDED and (not DO_PLOT_RETAINED):
                sample_hs = hs_p
                sample_tz = tz_p
                label_suffix = 'provided'
            if (not DO_PLOT_PROVIDED) and DO_PLOT_RETAINED:
                sample_hs = hs_r
                sample_tz = tz_r
                label_suffix = 'retained'
            if DO_PLOT_PROVIDED and DO_PLOT_RETAINED:
                sample_hs = np.concatenate([hs_p, hs_r])
                sample_tz = np.concatenate([tz_p, tz_r])
                label_suffix = 'provided and retained'

            contours_hs = []
            contours_tz = []
            for j in range(2):
                if j == 0:
                    return_period = 1
                else:
                    return_period = 20
                participant_nr = contribution_id
                if 11 >= contribution_id >= 9:
                    participant_nr = 9
                elif contribution_id > 11:
                    # Because contribution 9 holds 3 sets of contours.
                    participant_nr = contribution_id - 2
                folder_name = 'results/exercise-1/contribution-' + str(participant_nr)
                file_name = folder_name + '/' + \
                            lastname_firstname[contribution_id - 1] + '_dataset_' + \
                            dataset_char + '_' + str(return_period) + '.txt'
                (hs, tz) = read_contour(file_name)
                if participant_nr in (8, 9):
                    (tz, hs) = read_contour(file_name)
                contours_hs.append(hs)
                contours_tz.append(tz)

                hs_outside, tz_outside, hs_inside, tz_inside = \
                    points_outside(contours_hs[j],
                                   contours_tz[j],
                                   np.asarray(sample_hs),
                                   np.asarray(sample_tz))
                # Plot the data and the contour.
                if j == 0:
                    fig_row = dataset_count // 3
                    fig_col = dataset_count % 3
                    axs[fig_row, fig_col].set_title('Dataset ' + dataset_char +
                                                    ', ' + label_suffix)
                    print('Dataset ' + dataset_char + ', points outside the 1-yr Hs-Tz contour: ' +
                          str(len(hs_outside)))
                    plot_contour(contours_tz[j], contours_hs[j],
                                 ax=axs[fig_row, fig_col],
                                 contour_label=str(return_period) + ' year',
                                 style='b--')
                else:
                    print('Dataset ' + dataset_char + ', points outside the 20-yr Hs-Tz contour: ' +
                          str(len(hs_outside)))
                    sample_plot_data = SamplePlotData(x=np.asarray(sample_tz),
                                                    y=np.asarray(sample_hs),
                                                    ax=axs[fig_row, fig_col],
                                                    x_inside=tz_inside,
                                                    y_inside=hs_inside,
                                                    x_outside=tz_outside,
                                                    y_outside=hs_outside,
                                                    return_period=return_period)
                    plot_contour(contours_tz[j], contours_hs[j],
                                 ax=axs[fig_row, fig_col],
                                 contour_label=str(return_period) + ' years',
                                 x_label=label_tz.capitalize(),
                                 y_label=label_hs.capitalize(),
                                 x_lim=[0, 20],
                                 upper_ylim=15,
                                 style='b-',
                                 sample_plot_data=sample_plot_data)
            dataset_count = dataset_count + 1
        else:
            v_p, hs_p, label_v, label_hs = read_ecbenchmark_dataset(file_name_provided)
            v_r, hs_r, label_v, label_hs = read_ecbenchmark_dataset(file_name_retained)

            if DO_PLOT_PROVIDED and (not DO_PLOT_RETAINED):
                sample_v = v_p
                sample_hs = hs_p
                label_suffix = 'provided'
            if (not DO_PLOT_PROVIDED) and DO_PLOT_RETAINED:
                sample_v = v_r
                sample_hs = hs_r
                label_suffix = 'retained'
            if DO_PLOT_PROVIDED and DO_PLOT_RETAINED:
                sample_v = np.concatenate([v_p, v_r])
                sample_hs = np.concatenate([hs_p, hs_r])
                label_suffix = 'provided and retained'

            contours_v = []
            contours_hs = []
            for j in range(2):
                if j == 0:
                    return_period = 1
                else:
                    return_period = 50
                participant_nr = contribution_id
                if 11 >= contribution_id >= 9:
                    participant_nr = 9
                elif contribution_id > 11:
                    # Because contribution 9 holds 3 sets of contours.
                    participant_nr = contribution_id - 2
                folder_name = 'results/exercise-1/contribution-' + str(participant_nr)
                file_name = folder_name + '/' + \
                            lastname_firstname[contribution_id - 1] + '_dataset_' + \
                            dataset_char + '_' + str(return_period) + '.txt'
                if participant_nr in (1, 2, 3, 5, 6, 8, 10):
                    (hs, v) = read_contour(file_name)
                else:
                    (v, hs) = read_contour(file_name)
                contours_v.append(v)
                contours_hs.append(hs)

                v_outside, hs_outside, v_inside, hs_inside = \
                    points_outside(contours_v[j],
                                   contours_hs[j],
                                   np.asarray(sample_v),
                                   np.asarray(sample_hs))

                # Plot the data and the contour.
                if j == 0:
                    fig_row = dataset_count // 3
                    fig_col = dataset_count % 3
                    axs[fig_row, fig_col].set_title('Dataset ' + dataset_char +
                                                    ', ' + label_suffix)
                    print('Dataset ' + dataset_char + ', points outside the 1-yr V-Hs contour: ' +
                          str(len(hs_outside)))
                    plot_contour(contours_v[j], contours_hs[j],
                                 ax=axs[fig_row, fig_col],
                                 contour_label=str(return_period) + ' year',
                                 style='b--')
                else:

                    print('Dataset ' + dataset_char + ', points outside the 50-yr V-Hs contour: ' +
                          str(len(hs_outside)))
                    sample_plot_data = SamplePlotData(x=np.asarray(sample_v),
                                                    y=np.asarray(sample_hs),
                                                    ax=axs[fig_row, fig_col],
                                                    x_inside=v_inside,
                                                    y_inside=hs_inside,
                                                    x_outside=v_outside,
                                                    y_outside=hs_outside,
                                                    return_period=return_period)
                    plot_contour(contours_v[j], contours_hs[j],
                                 ax=axs[fig_row, fig_col],
                                 contour_label=str(return_period) + ' years',
                                 x_label=label_v.capitalize(),
                                 y_label=label_hs.capitalize(),
                                 x_lim=[0, 35],
                                 upper_ylim=20,
                                 style='b-',
                                 sample_plot_data=sample_plot_data)
            dataset_count = dataset_count + 1
    fig.tight_layout(pad=3.0)
    plt.suptitle(legends_for_contribution[contribution_id - 1])
    plt.show()
    fig_name_suffix = label_suffix.replace(" ", "_")
    fig.savefig('results/exercise-1/gfx/e1_contribution_' + str(contribution_id) + '_' +
                fig_name_suffix, dpi=150)
