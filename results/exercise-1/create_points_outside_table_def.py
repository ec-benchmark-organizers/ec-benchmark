import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.contour_analysis import points_outside

def mean_and_individuals_string(nr_outside):
    rstring = "{:.1f}".format(np.mean(nr_outside)) + ' (' + str(int(nr_outside[0])) + ', ' + \
              str(int(nr_outside[1])) + ', ' + str(int(nr_outside[2])) + ')'
    return rstring

all_rows = [['Contribution',
             'Num. points outside 1-yr contour',
             'Expected num. points outside 1-yr',
             'Num. points outside 1-yr where v>1 m/s and hs>1 m',
             'Num. points outside 50-yr contour',
             'Expected num. points outside 50-yr',
             'Num. points outside 20-yr where v>1 m/s and hs>1 m']]
expected_outside_1yr =  ['20', 'ca. 492', '?', '50', 'ca. 492', '492', '492', '492', 'ca. 492', 'ca. 492', '492']
expected_outside_20yr = ['1', 'ca. 12', '?', '1', 'ca. 12', '12', '12', '12', 'ca. 12', 'ca. 12', '12']


# Set these constant to choose if the provided, the retained or both datasets
# should be plotted together with the contours.
DO_USE_PROVIDED = True
DO_USE_RETAINED = True

dataset_chars = ['D', 'E', 'F']
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

for i in range(11):
    contribution_id = i + 1

    print('Starting analysis for ' + legends_for_contribution[contribution_id - 1] +
          ' (' + lastname_firstname[contribution_id -1] + ')')

    outside_1yr = np.empty((3, 1))
    outside_withthreshold_1yr = np.empty((3, 1))
    outside_50yr = np.empty((3, 1))
    outside_withthreshold_50yr = np.empty((3, 1))
    dataset_count = 0
    for dataset_char in dataset_chars:
        file_name_provided = 'datasets/' + dataset_char + '.txt'
        file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'

        v_p, hs_p, label_v, label_hs = read_ecbenchmark_dataset(file_name_provided)
        v_r, hs_r, label_v, label_hs = read_ecbenchmark_dataset(file_name_retained)

        if DO_USE_PROVIDED and (not DO_USE_RETAINED):
            sample_v = v_p
            sample_hs = hs_p
            label_suffix = 'provided'
        if (not DO_USE_PROVIDED) and DO_USE_RETAINED:
            sample_v = v_r
            sample_hs = hs_r
            label_suffix = 'retained'
        if DO_USE_PROVIDED and DO_USE_RETAINED:
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
            if contribution_id > 9:
                participant_nr = 9
            else:
                participant_nr = contribution_id
            folder_name = 'results/exercise-1/contribution-' + str(participant_nr)
            file_name = folder_name + '/' + \
                        lastname_firstname[contribution_id - 1] + '_dataset_' + \
                        dataset_char + '_' + str(return_period) + '.txt'
            if participant_nr in (1, 2, 3, 5, 6, 8):
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

            if j == 0:
                print('Dataset ' + dataset_char + ', points outside the 1-yr V-Hs contour: ' +
                      str(len(hs_outside)) + ', above threshold: ' +
                      str(len(hs_outside[(hs_outside > 1) & (v_outside > 1)])))
                outside_1yr[dataset_count] = len(hs_outside)
                outside_withthreshold_1yr[dataset_count] = \
                    len(hs_outside[(hs_outside > 1) & (v_outside > 1)])
            else:
                print('Dataset ' + dataset_char + ', points outside the 50-yr V-Hs contour: ' +
                      str(len(hs_outside)) + ', above threshold: ' +
                      str(len(hs_outside[(hs_outside > 1) & (v_outside > 1)])))
                outside_50yr[dataset_count] = len(hs_outside)
                outside_withthreshold_50yr[dataset_count] = \
                    len(hs_outside[(hs_outside > 1) & (v_outside > 1)])
        dataset_count = dataset_count + 1

    if contribution_id == 9:
        contr_abbrevation = '9 DS'
    elif contribution_id == 10:
        contr_abbrevation = '9 DS s.'
    elif contribution_id == 11:
        contr_abbrevation = '9 IFORM'
    else:
        contr_abbrevation = str(contribution_id)

    new_row = [contr_abbrevation,
                #mean_and_individuals_string(outside_1yr),
                "{:.1f}".format(np.mean(outside_1yr)),
                expected_outside_1yr[contribution_id - 1],
                #mean_and_individuals_string(outside_withthreshold_1yr),
                "{:.1f}".format(np.mean(outside_withthreshold_1yr)),
                mean_and_individuals_string(outside_50yr),
                expected_outside_20yr[contribution_id - 1],
                mean_and_individuals_string(outside_withthreshold_50yr)]
    all_rows = np.vstack([all_rows, new_row])

print('Tabulate Table:')
print(tabulate(all_rows, headers='firstrow'))

print('\nTabulate Latex:')
print(tabulate(all_rows, headers='firstrow', tablefmt='latex'))