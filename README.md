# A Benchmarking Exercise on Estimating Extreme Environmental Conditions: Methodology & Baseline Results
This is the official respository for the excercise annouced at OMAE 2019 [paper](mailto:?subject=Send%20benchmarking%20exercise%20paper&body=Hi%20Andreas,%0D%0A%0D%0Acould%20you%20send%20me%20a%20copy%20of%20the%20paper%20that%20describes%20the%20benchmarking%20exercise?%0D%0A%0D%0ABest%20regards,), [presentation](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/publications/2019-06-17_OMAE2019_BenchmarkingExercise.pdf)).
## Datasets
The 6 datasets are located in the folder [datasets](https://github.com/ec-benchmark-organizers/ec-benchmark/tree/master/datasets).

The original datasets are owned by NDBC (datasets A, B & C) and WDCC (dataset D, E & F) and their terms of use apply:
* NDBC: https://www.ndbc.noaa.gov
* WDCC: https://cera-www.dkrz.de/WDCC/ui/cerasearch/info?site=termsofuse
## Preparing your results
* To prepare your results, please provide the coordinates of the environmental contours using the same format that the datasets have. That means please separate all values with semicolons. The first line should contain the header. As an example, have a look at [this file](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/contour_coordinates/doe_john_dataset_a_1.txt). You can use the Python method [determine_file_name](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/read_write.py#L47) to determine the required file name and the method [write_contour](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/read_write.py#L104) to create the CSV file in the required format.
* In addition, we encourage you to plot your environmental contours using the plotting method provided in the file [organizers-code/plot.py](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/plot.py). If you are are not afluent in Python feel free to plot your contours with a different method.
## Entering the benchmark & submitting your results
To enter the benchmark, please submit the following files to <ecbenchmark@gmail.com>:
* Exercise 1 (environmental modeling and contour construction):
  * A title page (use [this form](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/EC_Benchmark_Exercise1.pdf); [here](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/EC_Benchmark_Exercise1_Example.pdf) is a filled example)
  * 12 CSV files, each containing the coordinates of one contour (example: [doe_john_dataset_a_1.txt
](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/contour_coordinates/doe_john_dataset_a_1.txt))
  * 6 figures, each showing one dataset with two contours (example: [doe_john_dataset_a_1_20.png](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/figures/doe_john_dataset_a_1_20.png))
  * Please compress these 19 files into a ZIP file and name it 'lastname_firstname_exercise1.zip' ([example](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/submission/doe_john_exercise1.zip)).
* Exercise 2 (uncertainty characterization):
  * A title page (use [this form](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/EC_Benchmark_Exercise2.pdf); [here](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/EC_Benchmark_Exercise2_Example.pdf) is a filled example)
  * 9 CSV files that together describe the three cases. Each case is described with three CSV files: one csv file should contain the median contour, one csv file the 2.5th percentile contour and one csv file the 97.5th percentile contour (example: [median](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/contour_coordinates/doe_john_years_1_median.txt), [2.5th percentile](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/contour_coordinates/doe_john_years_1_bottom.txt), [97.5th percentile](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/contour_coordinates/doe_john_years_1_upper.txt))
  * 6 figures, 3 of them for overlay plots (example:[doe_john_excercise2_5yr_allcontours.png](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/figures/doe_john_excercise2_5yr_allcontours.png)) and 3 of them for confidence interval plots (example: [doe_john_excercise2_5yr_confidenceintervals.png](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/figures/doe_john_excercise2_5yr_confidenceintervals.png))
  * Please compress these 16 files into a ZIP file and name it 'lastname_firstname_exercise2.zip' ([example](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/submission/doe_john_exercise2.zip)).
* You can either participate only in Exercise 1, only in Excercise 2 or in both.
## Questions
We intend to use GitHub issues like a forum. Feel free to [open an issue](https://github.com/ec-benchmark-organizers/ec-benchmark/issues/new) if you have a questions and we will answer it publicly in the same issue. Alternatively, you can contact us per email: <ecbenchmark@gmail.com>
