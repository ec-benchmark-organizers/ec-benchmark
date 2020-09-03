import matplotlib.pyplot as plt
import numpy as np

from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.plot import plot_confidence_interval, SamplePlotData

file_name = 'datasets/D.txt'
sample_v, sample_hs, label_v, label_hs = read_ecbenchmark_dataset(file_name)

lastname_firstname = ['GC_CGS', 'hannesdottir_asta',
                      'haselsteiner_andreas', 'vanem_DirectSampling']
style_for_participant = ['-r', '-g', '-k', '-c',]
legend_for_participant = ['Contribution 2',
                          'Contribution 3',
                          'Contribution 4',
                          'Contribution 9'
                          ]
contribution_nrs = [2, 3, 4, 9]
sample_lengths = [1, 5, 25]
contour_labels = ['50th percentile contour', '2.5th percentile contour',
                  '97.5th percentile contour']
ylims = [18, 18, 18]

for (sample_length,ylim) in zip(sample_lengths,ylims):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 8))
    for (i, ax1) in enumerate(ax.flatten()):
    # for i in range(4):
        contribution_nr = contribution_nrs[i]
        folder_name = 'results/exercise-2/contribution-' + str(contribution_nr)
        temp = folder_name + '/' + lastname_firstname[i] + '_years_' + str(sample_length)
        file_name_median = temp + '_median.txt'
        file_name_lower = temp + '_bottom.txt'
        file_name_upper = temp + '_upper.txt'
        v_median, hs_median = read_contour(file_name_median)
        v_lower, hs_lower = read_contour(file_name_lower)
        v_upper, hs_upper = read_contour(file_name_upper)

        # Plot the sample, the median contour and the confidence interval.
        # fig = plt.figure(figsize=(5, 5), dpi=150)
        # ax = fig.add_subplot(111)
        sample_plot_data = SamplePlotData(x=np.asarray(sample_v),
                                        y=np.asarray(sample_hs),
                                        ax=ax1,
                                        label='Dataset D')

        plot_confidence_interval(
            x_median=v_median, y_median=hs_median,
            x_bottom=v_lower, y_bottom=hs_lower,
            x_upper=v_upper, y_upper=hs_upper, ax=ax1,
            contour_labels=contour_labels, sample_plot_data=sample_plot_data)
        title_str = legend_for_participant[i]
        # if sample_length > 1:
        #     title_str = title_str + 's'
        ax1.set_title(title_str)
        ax1.set_xlabel(label_v.capitalize())
        ax1.set_ylabel(label_hs.capitalize())
        ax1.set_ylim(top=ylim)
        ax1.set_xlim(left=0, right=32)
        
    fig.tight_layout()
    fig.savefig('results/e2_samplelength_' + str(sample_length) + '.pdf', bbox_inches='tight')
    # plt.show()
