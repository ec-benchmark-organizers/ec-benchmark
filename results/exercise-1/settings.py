"""
Settings for plotting the results.

If you want to plot your contour too, you need to make the 
changes that are described in inline comments below.
"""
import numpy as np

lastname_firstname = [
    'Wei_Bernt',
    'GC_CGS',
    'hannesdottir_asta',
    'haselsteiner_andreas',
    'BV',
    'mackay_ed',
    'qiao_chi',
    'rode_anna',
    'vanem_DirectSampling',
    'vanem_DirectSamplingsmoothed',
    'vanem_IFORM',
    #'doe_john'
    # Add the filename prefix of your contribution here.
]
legends_for_contribution = [
    'Contribution 1',
    'Contribution 2',
    'Contribution 3',
    'Contribution 4',
    'Contribution 5',
    'Contribution 6',
    'Contribution 7',
    'Contribution 8',
    'Contribution 9, DS',
    'Contribution 9, DS smoothed',
    'Contribution 9, IFORM',
    #'John\'s new method'
    # Add a legend for your contribution here.
]

# Add an additional 1 for total exceedance and a 0 for marginal exceedance here.
# If your contribution is a total exceedance contour use a dash line to draw it, '--'
contour_classes = np.array([ 1,    0,   1,    1,    0,   0,   0,   0,   0,   0,   0,     ]) 
ls_for_contribution =      ['--', '-', '--', '--', '-', '-', '-', '-', '-', '-', '-', '-']

# Import Paired_12 instead of Paired_11 here.
from palettable.colorbrewer.qualitative import Paired_11 as mycorder
colors_for_contribution = np.array(mycorder.mpl_colors)
