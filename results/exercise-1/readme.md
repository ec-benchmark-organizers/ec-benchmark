# Applying the analysis to an additional contour method

You can use the provided scripts to apply the same analysis that was used in
the EC benchmark to some new environmental contours.

To do that you must
 - clone or fork this repository,
 - install the python packages listed in [results/requirements.txt](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/results/requirements.txt)
 - provide 1-year, 20-year and 50-year contour coordinates in the format that was specified for the EC benchmark for all datasets (A - F),
 - save these coordinates in the folder 'contribution-10' using the file_naming convention {last_name}_{first_name}_dataset_{dataset character}_{return period}.txt (the doe_john files are examples that you can overwrite, they are in the required format and follow the naming convention) and
 - adapt the file settings.py .

In settings.py
 - add the filename prefix to the variable 'lastname_firstname' (line 21),
 - add a legend for your environmental contours to the variable 'legends_for_contribution' (line 36),
 - add a 0 or 1 to 'contour_classes' depending upon which class your contour is (line 42) and
 - use 12 colors for plotting instead of 11 by changing the import (line 46)

 Then you can run the scripts that were used in the EC benchmark:
  - plot_e1_standardized.py (one plot for each contour method)
  - plot_benchmark_contours_dataset_abc.py (contour overlay)
  - plot_benchmark_contours_dataset_def.py (contour overlay)
  - plot_e1_maxima.py (maxima along the contour)
  - create_points_outside_table_abc.py (counts datapoints outside the contours)
  - create_points_outside_table_def.py (counts datapoints outside the contours)
  
The scripts must be run while you are in the repository's base folder (in "ec-benchmark").
From there run the scripts by typing, for example, 'python results/exercise-1/plot_benchmark_contours_dataset_abc.py'

## Video tutorial

[![Video](http://img.youtube.com/vi/TpXXtI6KRF8/0.jpg)](http://www.youtube.com/watch?v=TpXXtI6KRF8 "Video")
