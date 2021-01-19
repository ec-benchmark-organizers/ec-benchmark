# Applying the analysis to an additional contour method

You can use the provided scripts to apply the same analysis that was used in
the EC benchmark to some new environmental contours.

To do that you must
 - clone or fork this repository,
 - install the python packages listed in [results/requirements.txt](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/results/requirements.txt)
 - provide 1-year, 20-year and 50-year contour coordinates in the format that was specified for the EC benchmark (look at the other results to understand the format),
 - save these coordinates in the folder 'contribution-10' (the doe_john files
 are examples that you can overwrite) and
 - adapt the file settings.py .

In settings.py
 - add the filename prefix to the variable 'lastname_firstname',
 - add a legend for your environmental contours to the variable 'legends_for_contribution' and
 - add a 0 or 1 to 'contour_classes' depending upon which class your contour is.

 Then you can run the scripts that were used in the EC benchmark:
  - plot_e1_standardized.py (one plot for each contour method)
  - plot_benchmark_contours_dataset_abc.py (contour overlay)
  - plot_benchmark_contours_dataset_def.py (contour overlay)
  - plot_e1_maxima.py (maxima along the contour)
  - create_points_outside_table_abc.py (counts datapoints outside the contours)
  - create_points_outside_table_def.py (counts datapoints outside the contours)
  
The scripts must be run while you are in the repository's base folder in your shell (e.g. python results/exercise-1/plot_e1_standardized.py)
