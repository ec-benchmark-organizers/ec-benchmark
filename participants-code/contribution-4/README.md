# Repo for 'Global hierarchical models for wind and wave contours'

This is the repository for the paper 'Global hierarchical models for wind and
wave contours: Physical interpretations of the dependence functions' by A. F.
Haselsteiner, A. Sander, J.-H. Ohlendorf and K.-D. Thoben.

The results presented in this paper were also submitted to the benchmarking
exercise on estimating extreme environmental conditions (see DOI: 10.1115/OMAE2019-96523).

It contains the following things:
 * The used datasets (folder 'datasets')
 * Python files to reproduce the analysis
   * compute_contours_datasets_abc.py for the sea state models
   * compute_contours_datasets_def.py for the wind-wave state models
 * The coordinates of the computed environmental contours (folder 'contour-coordinates')

## Download and use the repository
To download this repository type
```console
git clone https://github.com/ahaselsteiner/2020-paper-omae-hierarchical-models.git
```

Then install the required Python packages by typing
```console
pip install -r requirements.txt
```
(if you are using pip for installing packages)
