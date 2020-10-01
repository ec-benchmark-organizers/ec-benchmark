import matplotlib.pyplot as plt
import numpy as np

from viroconcom.read_write import read_ecbenchmark_dataset, read_contour
from viroconcom.plot import plot_confidence_interval, SamplePlotData

fs = 12

file_name = 'datasets/D.txt'
sample_v, sample_hs, label_v, label_hs = read_ecbenchmark_dataset(file_name)

names = ['GC_CGS', 'hannesdottir_asta',
                      'haselsteiner_andreas', 'vanem_DirectSampling']
styles = ['-r', '-g', '-k', '-c',]
leg_strs = ['Contribution 2',
                          'Contribution 3',
                          'Contribution 4',
                          'Contribution 9']
nums = [2, 3, 4, 9]

prcntl_strs = ['50th percentile contour', '2.5th percentile contour',
                  '97.5th percentile contour']

slengths = [1, 5, 25]
ylims = [18, 18, 18]

fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(8.5,11))


for idx, (axi,name,style,leg_str,num) in enumerate(zip(axs, names, styles, leg_strs, nums)):
    for (ax,slength,ylim) in zip(axi, slengths, ylims):
        folder_name = 'results/exercise-2/contribution-' + str(num)
        temp = folder_name + '/' + name + '_years_' + str(slength)
        file_name_median = temp + '_median.txt'
        file_name_lower = temp + '_bottom.txt'
        file_name_upper = temp + '_upper.txt'
        v_median, hs_median = read_contour(file_name_median)
        v_lower, hs_lower = read_contour(file_name_lower)
        v_upper, hs_upper = read_contour(file_name_upper)
        
        sample_plot_data = SamplePlotData(x=np.asarray(sample_v),
                                          y=np.asarray(sample_hs),
                                          ax=ax,
                                          label='Dataset D')
        
        plot_confidence_interval(x_median=v_median, y_median=hs_median,
                                 x_bottom=v_lower, y_bottom=hs_lower,
                                 x_upper=v_upper, y_upper=hs_upper, ax=ax,
                                 contour_labels=prcntl_strs,
                                 sample_plot_data=sample_plot_data)


        lgd = ax.get_legend()
        lgd.remove()

# titles along top
for (ax, slength) in zip(axs[0], slengths):
    if slength is 1:
        tstr = 'Using {}-yr of data'.format(slength)
    else:
        tstr = 'Using {}-yrs of data'.format(slength)
    ax.set_title(tstr)


for (ax, num) in zip(axs[:,0], nums):
    ax.set_ylabel('Contrib. {}'.format(num))

for ax in axs.flat:
    # ax.tick_params(axis='both', which='major', labelsize=fs)   
    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.label_outer()
    ax.set_ylim(top=ylim)
    ax.set_xlim(left=0, right=32)

fig.tight_layout(rect=(0.075,0.1,1,1))

fig.text(0.5, 0.075, label_hs.capitalize(), 
         ha='center', 
         fontsize=fs, 
         weight='bold')
fig.text(0.04, 0.5, label_v.capitalize(), 
         va='center', 
         rotation='vertical', 
         fontsize=fs, 
         weight='bold')

lgd = fig.legend(axs[0,0].get_legend_handles_labels()[0], 
           axs[0,0].get_legend_handles_labels()[1],
           loc = 8, 
           ncol=2, 
           fontsize = fs, 
           bbox_to_anchor=(0.5,0))

fig.savefig('results/e2.pdf',
            pad_inches=0.35,
            bbox_extra_artists=[lgd])
