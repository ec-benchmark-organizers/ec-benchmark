import numpy as np
import matplotlib as mpl


def points_outside(contour_x, contour_y, x, y):
    """
    Determines the observations outside the region enclosed by the contour.

    Parameters
    ----------
    contour_x : ndarray of doubles
        The contour's coordinates in the x-direction.
    contour_y : ndarray of doubles
        The contour's coordiantes in the y-direction.
    x : ndarray of doubles
        The sample's first environmental variable.
    y : ndarray of doubles
        The sample's second environmental variable.

    Returns
    -------
    outside_x : nparray
        The observations that are outside of the contour of variable 1.

    outside_y : nparray
        The observations that are outside of the contour of variable 2.

    """
    contour_path = mpl.path.Path(np.column_stack((contour_x, contour_y)))
    is_inside_contour = contour_path.contains_points(np.column_stack((x, y)))
    indices_is_outside = np.argwhere(is_inside_contour == False)
    x_outside = x[indices_is_outside]
    y_outside =  y[indices_is_outside]
    indices_is_inside = np.argwhere(is_inside_contour == True)
    x_inside = x[indices_is_inside]
    y_inside =  y[indices_is_inside]

    return (x_outside, y_outside, x_inside, y_inside)