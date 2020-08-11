import matplotlib.pyplot as plt

from viroconcom.read_write import read_ecbenchmark_dataset


dataset_chars = ['A', 'B', 'C', 'D', 'E', 'F']
figp, axsp = plt.subplots(2, 3, figsize=(16, 10))
figr, axsr = plt.subplots(2, 3, figsize=(16, 10))

i = 0
for dataset_char in dataset_chars:
    file_name_provided = 'datasets/' + dataset_char + '.txt'
    file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'
    fig_row = i // 3
    fig_col = i % 3
    axsp[fig_row, fig_col].set_title('Dataset ' + dataset_char + ', provided')
    axsr[fig_row, fig_col].set_title('Dataset ' + dataset_char + ', retained')

    if dataset_char in ('A', 'B', 'C'):
        hs_p, tz_p, label_hs, label_tz = read_ecbenchmark_dataset(file_name_provided)
        hs_r, tz_r, label_hs, label_tz = read_ecbenchmark_dataset(file_name_retained)
        axsp[fig_row, fig_col].scatter(tz_p, hs_p, c='black',alpha=0.5)
        axsp[fig_row, fig_col].set_xlabel(label_tz.capitalize())
        axsp[fig_row, fig_col].set_xlim([0, 20])
        axsp[fig_row, fig_col].set_ylim([0, 15])
        axsr[fig_row, fig_col].scatter(tz_r, hs_r, c='black', alpha=0.5)
        axsr[fig_row, fig_col].set_xlabel(label_tz.capitalize())
        axsr[fig_row, fig_col].set_xlim([0, 20])
        axsr[fig_row, fig_col].set_ylim([0, 15])
    else:
        v_p, hs_p, label_v, label_hs = read_ecbenchmark_dataset(file_name_provided)
        v_r, hs_r, label_v, label_hs = read_ecbenchmark_dataset(file_name_retained)
        axsp[fig_row, fig_col].scatter(v_p, hs_p, c='black', alpha=0.5)
        axsp[fig_row, fig_col].set_xlabel(label_v.capitalize())
        axsp[fig_row, fig_col].set_xlim([0, 35])
        axsp[fig_row, fig_col].set_ylim([0, 20])
        axsr[fig_row, fig_col].scatter(v_r, hs_r, c='black', alpha=0.5)
        axsr[fig_row, fig_col].set_xlabel(label_v.capitalize())
        axsr[fig_row, fig_col].set_xlim([0, 35])
        axsr[fig_row, fig_col].set_ylim([0, 20])
    axsp[fig_row, fig_col].set_ylabel(label_hs.capitalize())
    axsr[fig_row, fig_col].set_ylabel(label_hs.capitalize())
    axsp[fig_row, fig_col].spines['right'].set_visible(False)
    axsp[fig_row, fig_col].spines['top'].set_visible(False)
    axsp[fig_row, fig_col].yaxis.set_ticks_position('left')
    axsp[fig_row, fig_col].xaxis.set_ticks_position('bottom')
    axsr[fig_row, fig_col].spines['right'].set_visible(False)
    axsr[fig_row, fig_col].spines['top'].set_visible(False)
    axsr[fig_row, fig_col].yaxis.set_ticks_position('left')
    axsr[fig_row, fig_col].xaxis.set_ticks_position('bottom')
    i = i + 1

plt.show()
figp.savefig('datasets/datasets_provided', dpi=150)
figr.savefig('datasets-retained/datasets_retained', dpi=150)
