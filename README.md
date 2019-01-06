# A Benchmarking Exercise on Estimating Extreme Environmental Conditions for Engineering Design
This is the official respository for the excercise annouced at OMAE 2019 (**link to paper**).
## Datasets
The 6 datasets are located in the folder [datasets](https://github.com/ec-benchmark-organizers/ec-benchmark/tree/master/datasets).
## Preparing your results
* To prepare your results, please provide the coordinates of the environmental contours using the same format that the datasets have. That means please separate all values with semicolons. The first line should contain the header. As an example, have a look at [this file](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/doe_john_dataset_a_20.txt). You can use the Python method [determine_file_name](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/read_write.py#L38) to determine the required file name and the method [write_contour](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/read_write.py#L64) to create the CSV file in the required format.
* In addition, we encourage you to plot your environmental contours using the plotting method provided in the file [organizers-code/plot.py](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/plot.py). If you are are not afluent in Python feel free to plot your contours with a different method.
## Entering the benchmark & submitting your results
To enter the benchmark, please submit the following files to **not-yet-set@email.domain**:
* Exercise 1 (environmental modeling and contour construction):
  * A title page (use [this form](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/EC_Benchmark_Exercise1.pdf); [here](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/EC_Benchmark_Exercise1_Example.pdf) is a filled example)
  * 12 CSV files, each containing the coordinates of one contour (example: [doe_john_dataset_a_20.txt
](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/doe_john_dataset_a_20.txt))
  * 12 figures, each showing one contour (example: [doe_john_dataset_a_20.png](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/organizers-code/doe_john_dataset_a_20.png))
  * Please compress these 25 files into a ZIP file and name it 'lastname_firstname_exercise1.zip' (example: **not yet**).
* Exercise 2 (uncertainty characterization):
  * A title page (use [this form](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/EC_Benchmark_Exercise2.pdf); [here](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/EC_Benchmark_Exercise2_Example.pdf) is a filled example)
  * 9 CSV files that together describe 3 contours. Each contour is described with 3 CSV files: one csv file should contain the most-likely contour, one csv file the bottom confidence interval and one csv file the upper confidence interval.
  * 3 figures, each showing one contour (example: **not yet**)
  * Please compress these 13 files into a ZIP file and name it 'lastname_firstname_exercise2.zip' (example: **not yet**).
* You can either participate only in Exercise 1, only in Excercise 2 or in both.
## Questions
We intend to use GitHub issues like a forum. Feel free to [open an issue](https://github.com/ec-benchmark-organizers/ec-benchmark/issues/new) if you have a questions and we will answer it publicly in the same issue. Alternatively, you can contact us per email: **not-yet-set@email.domain**
