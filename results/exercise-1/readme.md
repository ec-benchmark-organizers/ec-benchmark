# Applying the analysis to an additional contour method

You can use the provided scripts to apply the same analysis that was used in
the EC benchmark to some new environmental contours.

To do that you must
 - clone / fork this repository,
 - provide contour coordinates in the format that was specified for the EC benchmark,
 - save these coordinates in the folder 'contribution-10' (the doe_john files
 are examples that you can overwrite) and
 - adapt the file settings.py .

In settings.py
 - add the filename prefix to the variable 'lastname_firstname' and
 - add a legend for your environmental contours to the variable 'legends_for_contribution'.

 Then you can run the scripts that were used in the EC benchmark:
  - plot_e1_standardized.py (one plot for each contour method)
  - plot_benchmark_contours_dataset_abc.py (contour overlay)
  - plot_benchmark_contours_dataset_def.py (contour overlay)
  - plot_e1_maxima.py (maxima along the contour)
  - create_points_outside_table_abc.py (counts datapoints outside the contours)
  - create_points_outside_table_def.py (counts datapoints outside the contours)
