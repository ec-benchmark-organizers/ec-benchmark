import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from palettable.colorbrewer.qualitative import Paired_6 as mycorder


from viroconcom.read_write import read_ecbenchmark_dataset


dataset_chars = ['A', 'B', 'C', 'D', 'E', 'F']
dataset_colors = mycorder.mpl_colors
n_lags = 14 * 24 # Two weeks.
c_hs = np.empty((6, n_lags + 1))
c_v = np.empty((3, n_lags + 1))
lag = np.arange(0, n_lags + 1, 1)
lag = lag / 24
for i, dataset_char in enumerate(dataset_chars):
    file_name_provided = 'datasets/' + dataset_char + '.txt'
    if dataset_char in ['A', 'B', 'C']:
        hs_p, tz_p, label_hs, label_tz = read_ecbenchmark_dataset(file_name_provided)
        c_hs[i] = sm.tsa.acf(hs_p, nlags=n_lags)
    else:
        v_p, hs_p, label_v, label_hs = read_ecbenchmark_dataset(file_name_provided)
        c_hs[i] = sm.tsa.acf(hs_p, nlags=n_lags)
        c_v[i - 3] = sm.tsa.acf(v_p, nlags=n_lags)

fig, ax = plt.subplots(1, 2, sharex='row', sharey='row', figsize=(9, 4))
for i in range(6):
    ax[0].plot(lag, c_hs[i], label='Dataset ' + dataset_chars[i], color=dataset_colors[i])
for i in range(3):
    ax[1].plot(lag, c_v[i], color=dataset_colors[i + 3])
    
for i in range(2):
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].yaxis.set_ticks_position('left')
    ax[i].xaxis.set_ticks_position('bottom')

lgd = fig.legend(loc='lower center',
                 ncol=6,
                 prop={'size': 8})
ax[0].set_xlabel('Lag (days)')
ax[0].set_ylabel('Autocorrelation of $H_s$')
ax[1].set_xlabel('Lag (days)')
ax[1].set_ylabel('Autocorrelation of $U_{10}$')
fig.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()
fig.savefig('results/discussion/autocorrelation.pdf')
