import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from palettable.colorbrewer.qualitative import Paired_6 as mycorder
import scipy.signal as ss

from viroconcom.read_write import read_ecbenchmark_dataset


dataset_chars = ['A', 'B', 'C', 'D', 'E', 'F']
dataset_colors = mycorder.mpl_colors
n_lags = 14 * 24 # Two weeks.
c_hs_buoy = np.empty((3, n_lags + 1))
c_tz = np.empty((3, n_lags + 1))
c_hs_hincast = np.empty((3, n_lags + 1))
c_v = np.empty((3, n_lags + 1))
lag = np.arange(0, n_lags + 1, 1)
lag = lag / 24
hs_buoy_list = []
tz_list = []
hs_hindcast_list = []
v_list = []
for i, dataset_char in enumerate(dataset_chars):
    file_name_provided = 'datasets/' + dataset_char + '.txt'
    if dataset_char in ['A', 'B', 'C']:
        hs_p, tz_p, label_hs, label_tz = read_ecbenchmark_dataset(file_name_provided)
        c_hs_buoy[i] = sm.tsa.acf(hs_p, nlags=n_lags)
        c_tz[i - 3] = sm.tsa.acf(tz_p, nlags=n_lags)
        hs_buoy_list.append(hs_p)
        tz_list.append(tz_p)
    else:
        v_p, hs_p, label_v, label_hs = read_ecbenchmark_dataset(file_name_provided)
        c_hs_hincast[i - 3] = sm.tsa.acf(hs_p, nlags=n_lags)
        c_v[i - 3] = sm.tsa.acf(v_p, nlags=n_lags)
        hs_hindcast_list.append(hs_p)
        v_list.append(v_p)


def ccf(x, y, lag_max = 100):
    # Calculates the cross correlation of x and y.
    # Thanks to https://stackoverflow.com/questions/53959879/how-do-i-get-rs-ccf-in-python
    result = ss.correlate(y - np.mean(y), x - np.mean(x), method='direct') / (np.std(y) * np.std(x) * len(y))
    length = (len(result) - 1) // 2
    lo = length - lag_max
    hi = length + (lag_max + 1)

    return result[lo:hi]

fig, ax = plt.subplots(2, 3, sharex='row', sharey='row', figsize=(9, 6))
for i in range(3):
    ax[0,0].plot(lag, c_hs_buoy[i], label='Dataset ' + dataset_chars[i], color=dataset_colors[i])
    ax[0,1].plot(lag, c_tz[i], color=dataset_colors[i])
    c = ccf(hs_buoy_list[i], tz_list[i], lag_max=n_lags)
    ax[0,2].plot(lag, c[n_lags:], color=dataset_colors[i])
    print(r'Maximum auto-correlation when \tau = hours ')
    print(np.argmax(c[n_lags:]))
for i in range(3):
    ax[1,0].plot(lag, c_hs_hincast[i], label='Dataset ' + dataset_chars[i + 3], color=dataset_colors[i + 3])
    ax[1,1].plot(lag, c_v[i], color=dataset_colors[i + 3])
    c = ccf(v_list[i], hs_hindcast_list[i], lag_max=n_lags)
    ax[1,2].plot(lag, c[n_lags:], color=dataset_colors[i + 3])
    print(r'Maximum auto-correlation when \tau = hours ')
    print(np.argmax(c[n_lags:]))

for i in range(2):
    for j in range(3):
        ax[i,j].spines['right'].set_visible(False)
        ax[i,j].spines['top'].set_visible(False)
        ax[i,j].yaxis.set_ticks_position('left')
        ax[i,j].xaxis.set_ticks_position('bottom')

lgd = fig.legend(loc='lower center',
                 ncol=6,
                 prop={'size': 8})
ax[0,0].set_ylabel('Auto-correlation of $H_s$')
ax[0,1].set_ylabel('Auto-correlation of $T_z$')
ax[0,2].set_ylabel(r'Correlation of $H_s(t)$ and $T_z(t + \tau)$')
ax[1,0].set_ylabel('Auto-correlation of $H_s$')
ax[1,1].set_ylabel('Auto-correlation of $U_{10}$')
ax[1,2].set_ylabel(r'Correlation of $U_{10}(t)$ and $H_s(t + \tau)$')
ax[1,0].set_xlabel('Lag (days)')
ax[1,1].set_xlabel('Lag (days)')
ax[1,2].set_xlabel('Lag (days)')
fig.tight_layout(rect=(0, 0.05, 1, 1))
plt.show()
fig.savefig('results/metocean/gfx/autocorrelation.pdf')
