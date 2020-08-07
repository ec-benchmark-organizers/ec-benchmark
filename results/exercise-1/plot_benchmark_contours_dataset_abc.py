import matplotlib.pyplot as plt
import numpy as np

from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.plot import plot_contour

dataset_char = 'C'
return_period = 20

file_name = 'datasets/' + dataset_char + '.txt'
sample_hs, sample_tz, label_hs, label_tz = read_ecbenchmark_dataset(file_name)

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
contour_hs_1 = []
contour_tz_1 = []
max_hs_on_contour = np.empty(n_contours_to_analyze)
for i in range(n_contours_to_analyze):
    participant_nr = i + 1
    if participant_nr > 9:
        participant_nr = 9
    folder_name = 'results/exercise-1/participant-' + str(participant_nr)
    file_name = folder_name + '/' + lastname_firstname[i] + '_dataset_' + \
                dataset_char + '_' + str(return_period) + '.txt'
    (hs, tz) = read_contour(file_name)
    if i in (7, 8, 9, 10):
        (tz, hs) = read_contour(file_name)
    contour_hs_1.append(hs)
    contour_tz_1.append(tz)
    max_hs_on_contour[i] = max(hs[~np.isnan(tz)])

# Plot the data and the contour.
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
plt.scatter(sample_tz, sample_hs, c='black', alpha=0.5)
for i in range(n_contours_to_analyze):
    ylim = max([1.05 * max(max_hs_on_contour), max(sample_hs)])
    plot_contour(contour_tz_1[i], contour_hs_1[i],
                 ax=ax, x_label=label_tz.capitalize(), y_label=label_hs.capitalize(),
                 line_style=style_for_participant[i],
                 contour_label=legend_for_participant[i],
                 upper_ylim=ylim)
    plt.legend(prop={'size': 6})
    plt.title('Dataset ' + dataset_char + ', ' + str(return_period) + '-year contour')
plt.show()