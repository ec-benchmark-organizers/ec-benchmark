import numpy as np
import csv


def read_dataset(path='../datasets/A.txt'):
    """
    Reads a datasets provided for the environmental contour benchmark.

    Parameters
    ----------
    path : string
        Path to dataset including the file name, defaults to '../datasets/A.txt'
    Returns
    -------
    x : ndarray of doubles
        Observations of the environmental variable 1.
    y : ndarray of doubles
        Observations of the environmental variable 2.
    x_label : str
        Label of the environmantal variable 1.
    y_label : str
        Label of the environmental variable 2.

    """

    x = list()
    y = list()
    x_label = None
    y_label = None
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        idx = 0
        for row in reader:
            if idx == 0:
                x_label = row[1][1:] # Ignore first char (is a white space).
                y_label = row[2][1:] # Ignore first char (is a white space).
            if idx > 0: # Ignore the header
                x.append(float(row[1]))
                y.append(float(row[2]))
            idx = idx + 1

    x = np.asarray(x)
    y = np.asarray(y)
    return (x, y, x_label, y_label)


def determine_file_name_e1(first_name, last_name, dataset, return_period):
    """
    Returns the file name for Excercise 1 as required by the benchmark's rules.

    Parameters
    ----------
    first_name : str
        Corresponding author's first name.
    last_name: str
        Corresponding author's last name.
    dataset : char
        Must be an element of {'A', 'B', 'C', 'D', 'E', 'F'} (or small caps)
    return_period : int
        The contour's return period in years.


    Returns
    -------
    file_name : str
        File name as required by the benchmark's rules.
    """
    file_name = last_name + '_' + first_name + '_dataset_' + dataset + \
                 '_' + str(return_period) + '.txt'
    return file_name.lower()


def determine_file_name_e2(first_name, last_name, bootstrap_years, type):
    """
    Returns the file name for Exercise 2 as required by the benchmark's rules.

    Parameters
    ----------
    first_name : str
        Corresponding author's first name.
    last_name: str
        Corresponding author's last name.
    bootstrap_years : int
        Number of years of the bootstrap sample.
    type : str
        Must be an element of {'median', 'bottom', 'upper'}

    Returns
    -------
    file_name : str
        File name as required by the benchmark's rules.
    """
    allowable_types = ['median', 'bottom', 'upper']
    if (type in allowable_types):
        file_name = last_name + '_' + first_name + '_years_' \
                    + str(bootstrap_years) + '_' + type + '.txt'
    else:
        raise ValueError('The parameter type must be one of the following '
                         'strings: ' + str(allowable_types) + '. However, it was: '
                         + str(type))
    return file_name.lower()


def write_contour(x, y, path, label_x='significant wave height [m]', label_y='zero-up-crossing period [s]'):
    """
    Writes the contour coordinates in the required format.

    Parameters
    ----------
    x : ndarray of doubles
        Values in the first dimensions of the contour's coordinates.
    y : ndarray of doubles
        Values in the second dimensions of the contour's coordinates.
    path : string
        Path including folder and file name where the contour should be saved.
    label_x : str
        Name and unit of the first environmental variable,
        defaults to 'significant wave height [m]'.
    label_y : str
        Name and unit of the second environmental variable,
        defaults to 'zero-up-crossing period [s]'.

    """
    with open(path, mode='w', newline='') as contour_file:
        contour_writer = csv.writer(contour_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        contour_writer.writerow([label_x, label_y])
        for xi,yi in zip(x,y):
            contour_writer.writerow([str(xi), str(yi)])


def read_contour(path):
    """
    Reads a datasets provided for the environmental contour benchmark.

    Parameters
    ----------
    path : string
        Path to contour including the file name.
    Returns
    -------
    x : ndarray of doubles
        Observations of the environmental variable 1.
    y : ndarray of doubles
        Observations of the environmental variable 2.

    """

    x = list()
    y = list()
    with open(path, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=';')
        idx = 0
        for row in reader:
            if idx > 0: # Ignore the header
                x.append(float(row[0]))
                y.append(float(row[1]))
            idx = idx + 1

    x = np.asarray(x)
    y = np.asarray(y)
    return (x, y)