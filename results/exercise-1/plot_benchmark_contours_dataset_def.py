import matplotlib.pyplot as plt
import numpy as np

from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.plot import plot_contour

dataset_chars = ['D', 'E', 'F']
return_periods = [1, 50]
lastname_firstname = ['Wei_Bernt', 'GC_CGS', 'hannesdottir_asta',
                      'haselsteiner_andreas', 'BV', 'mackay_ed',
                      'qiao_chi', 'rode_anna', 'vanem_DirectSampling',
                      'vanem_DirectSamplingsmoothed', 'vanem_IFORM']
style_for_participant = ['-b', '-r', '-g', '-k', '--b', '--r', '--g', '--k', '-c', '--c', '.-c']
legend_for_participant = ['Participant 1',
                          'Participant 2',
                          'Participant 3',
                          'Participant 4',
                          'Participant 5',
                          'Participant 6',
                          'Participant 7',
                          'Participant 8',
                          'Participant 9, DS',
                          'Participant 9, DS smoothed',
                          'Participant 9, IFORM'
                          ]
n_contours_to_analyze = 11

for dataset_char in dataset_chars:
    for return_period in return_periods:
        # Load the environmental data.
        file_name = 'datasets/' + dataset_char + '.txt'
        sample_v, sample_hs, label_v, label_hs = read_ecbenchmark_dataset(file_name)

        contours_v = []
        contours_hs = []
        max_hs_on_contours = np.empty(n_contours_to_analyze)
        for i in range(n_contours_to_analyze):
            participant_nr = i + 1
            if participant_nr > 9:
                participant_nr = 9
            folder_name = 'results/exercise-1/participant-' + str(participant_nr)
            file_name = folder_name + '/' + lastname_firstname[i] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            if participant_nr in (1, 2, 3, 5, 6, 8):
                (hs, v) = read_contour(file_name)
            else:
                (v, hs) = read_contour(file_name)
            contours_v.append(v)
            contours_hs.append(hs)
            max_hs_on_contours[i] = max(hs[~np.isnan(hs)])

        # Plot the data and the contour.
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.scatter(sample_v, sample_hs, c='black', alpha=0.5)
        for i in range(n_contours_to_analyze):
            ylim = 1.05 * max([max(max_hs_on_contours), max(sample_hs)])
            plot_contour(contours_v[i], contours_hs[i],
                         ax=ax, x_label=label_v.capitalize(), y_label=label_hs.capitalize(),
                         line_style=style_for_participant[i],
                         contour_label=legend_for_participant[i],
                         upper_ylim=ylim)
            plt.legend(prop={'size': 6})
            plt.title('Dataset ' + dataset_char + ', ' + str(return_period) + '-year contour')
        #plt.show()
        fig.savefig('e1_overlay_dataset_' + dataset_char + '_returnperiod_' +
                    str(return_period), dpi=150)
