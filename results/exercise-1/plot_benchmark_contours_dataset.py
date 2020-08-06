import matplotlib.pyplot as plt
import numpy as np
from palettable.colorbrewer.qualitative import Paired_9 as mycorder
from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.plot import plot_contour
import os
import string


def plotAll(dataset_char, return_period, ylim=None, savefile=False):
    file_name = os.path.join('..','..','datasets', dataset_char + '.txt')
    
    
    if dataset_char in list(string.ascii_uppercase)[0:3]:
        sample_hs, sample_tz, label_hs, label_tz = read_ecbenchmark_dataset(file_name)
    elif dataset_char in list(string.ascii_uppercase)[3:6]:
        sample_tz, sample_hs, label_tz, label_hs = read_ecbenchmark_dataset(file_name)
    
    lastname_firstname = ['Wei_Bernt', 'GC_CGS', 'hannesdottir_asta',
                          'haselsteiner_andreas', 'BV', 'mackay_ed',
                          'qiao_chi', 'rode_anna', 'vanem_DirectSampling',
                          'vanem_DirectSamplingsmoothed', 'vanem_IFORM']
    style_for_participant = ['-', '-', '-', 
                             '-', '-', '-', 
                             '-', '-', '-', 
                             ':', '--']
    color_for_participant = mycorder.mpl_colors
    for idx in range(3):
        color_for_participant.append(color_for_participant[8])
    
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
        folder_name = 'participant-' + str(participant_nr)
        file_name = os.path.join(folder_name, lastname_firstname[i] + 
                                 '_dataset_' + dataset_char + '_' + str(return_period) + '.txt')
        print(file_name)
        file_name = folder_name + '/' + lastname_firstname[i] + '_dataset_' + dataset_char + '_' + \
                    str(return_period) + '.txt'
        
        if dataset_char in list(string.ascii_uppercase)[0:3]:
            if i in (7, 8, 9, 10):
                (tz, hs) = read_contour(file_name)
            else:
                (hs, tz) = read_contour(file_name)
        elif dataset_char in list(string.ascii_uppercase)[3:6]:
            if participant_nr in (1, 2, 3, 5, 8):
                (hs, tz) = read_contour(file_name)
            else:
                (tz, hs) = read_contour(file_name)
            
        contour_hs_1.append(hs)
        contour_tz_1.append(tz)
        max_hs_on_contour[i] = max(hs[~np.isnan(tz)])
    
    # Plot the data and the contour.
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    
    pds = plt.scatter(sample_tz, sample_hs, c='black', alpha=0.05, zorder=-20)
    pls = []
    for i in range(n_contours_to_analyze):
        pls.append(plot_contour(contour_tz_1[i], contour_hs_1[i],
                      ax=ax, 
                      x_label=label_tz.capitalize(), 
                      y_label=label_hs.capitalize(),
                      line_style=style_for_participant[i],
                      color=color_for_participant[i],
                      contour_label=legend_for_participant[i]))
        
        plt.legend(prop={'size': 6})
        plt.title('Dataset ' + dataset_char + ', ' + str(return_period) + '-year contour')
    
    # ylim = max([1.05 * max(max_hs_on_contour), max(sample_hs)])
    if ylim:
        ax.set_ylim(top=ylim)
    
    if savefile:
        fname = os.path.join('plots', 'exercise_1_' + dataset_char + 
                             '_' + str(return_period) + '.pdf')
        ax.set_rasterization_zorder(-10)
        fig.savefig(fname, dpi=250)


for dataset_char in list(string.ascii_uppercase)[0:3]:
    for return_period in [1, 20]:
        plotAll(dataset_char, return_period, ylim=14, savefile=True)

for dataset_char in list(string.ascii_uppercase)[3:6]:
    for return_period in [1, 50]:
        _ = plotAll(dataset_char, return_period, ylim=20, savefile=True)

# plt.show()
