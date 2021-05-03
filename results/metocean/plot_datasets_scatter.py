import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from palettable.colorbrewer.qualitative import Paired_9 as mycorder
from viroconcom.read_write import read_ecbenchmark_dataset





datasets_hstz = ['A', 'B', 'C']
datasets_vhs = ['D', 'E', 'F']
latitudes =  [43.525,   28.508,  25.897, 54.0,  55.0,  59.5 ]
longitudes = [-70.141, -80.185, -89.668, 6.575, 1.175, 4.325]


fig, axs = plt.subplots(2, 3, sharex='row', sharey='row', figsize=(10, 8))
max_hs_of_sample = 0
for i, ax0 in enumerate(axs):
    if i == 0:
        datasets = datasets_hstz
    else:
        datasets = datasets_vhs
    for j, (dataset_char, ax1) in enumerate(zip(datasets, ax0)):
        # Load the environmental data.
        file_name_provided = 'datasets/' + dataset_char + '.txt'
        file_name_retained = 'datasets-retained/' + dataset_char + 'r.txt'
        x1_p, x2_p, x1_label, x2_label = read_ecbenchmark_dataset(file_name_provided)
        x1_r, x2_r, x1_label, x2_label = read_ecbenchmark_dataset(file_name_retained)
        if i == 1:
            x1_p, x2_p = x2_p, x1_p
            x1_r, x2_r = x2_r, x1_r
            x1_label, x2_label = x2_label, x1_label

        max_hs_of_sample = max([max_hs_of_sample, max(x1_p), max(x1_r)])

        # Scatter plot
        ax1.scatter(x2_p, x1_p, c='black', alpha=0.5, zorder=-2)
        ax1.scatter(x2_r, x1_r, marker='v', facecolor='None',
                    edgecolor='black', alpha=0.5, zorder=-2)

        ax1.set_rasterization_zorder(-1)
        ax1.set_xlabel(x2_label.capitalize())
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Map inset
        axins = inset_axes(ax1, width="30%", height="30%", loc=2)
        latitude = latitudes[i * 3 + j]
        longitude = longitudes[i * 3 + j]
        if i == 0:
            width = 5E6
            height = 5E6
            lat_0 = 40
            lon_0 = -80
        else:
            width = 2E6
            height = 2E6
            lat_0 = 55
            lon_0 = 4
        m = Basemap(projection='lcc', resolution='l',
                    width=width, height=height, 
                    lat_0=lat_0, lon_0=lon_0,)
        m.fillcontinents(color="#FFDDCC", lake_color='white')

        # Map (long, lat) to (x, y) for plotting
        x, y = m(longitude, latitude)
        axins.plot(x, y, 'or', markersize=3)
        axins.annotate(dataset_char, m(longitude, latitude + 0.8 * height / 2E6), color='red', ha='center')


for ax in axs[:,0]:
    ax.set_ylabel(x1_label.capitalize())

fig.tight_layout(rect=(0,0.05,1,1))
plt.savefig('results/metocean/gfx/datasets_scatter.pdf', bbox_inches='tight')
plt.show()
