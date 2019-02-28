from shapely.geometry import LineString, Point, MultiPoint
import numpy as np
import matplotlib.pyplot as plt
import warnings

def contour_intersection(contour_x, contour_y, line_x, line_y, do_plot_in_case_of_failure=False):
    """
    Computes the intersection point between two lines.

    Parameters
    ----------
    contour_x : ndarray of floats
    contour_y : ndarray of floats
    line_x : ndarray of floats
    line_y : ndarray of floats
    do_plot_in_case_of_failure : Boolean,
        if True a plot of the two lines for which no intersection could be
        found is shown.
    Returns
    -------
    x : float
    y : float
    """

    point_list_l1 = list()
    for x,y in zip(contour_x, contour_y):
        point = Point(x, y)
        point_list_l1.append(point)
    contour = LineString(point_list_l1)

    point_list_l2 = list()
    for x,y in zip(line_x, line_y):
        point = Point(x, y)
        point_list_l2.append(point)
    line2 = LineString(point_list_l2)

    intersection = contour.intersection(line2)
    if type(intersection) is Point:
        return intersection.x, intersection.y
    if type(intersection) is MultiPoint:
        origin = Point(line_x[0], line_y[0])
        for i, p in enumerate(intersection.geoms):
            if i == 0:
                inter_x = p.x
                inter_y = p.y
                longest_distance = origin.distance(p)
            else:
                if origin.distance(p) > longest_distance:
                    inter_x = p.x
                    inter_y = p.y
        return inter_x, inter_y
    else:
        print('The result is: ' + str(intersection))
        warnings.warn('No point of intersection could be found. '
                      'Returning (nan, nan).', UserWarning)
        if do_plot_in_case_of_failure:
            fig = plt.figure(figsize=(5, 5), dpi=150)
            fig.add_subplot(111)
            plt.plot(contour_x, contour_y, 'b-')
            plt.plot(line_x, line_y, 'r-')
            plt.show()
        return np.nan, np.nan


if __name__ == '__main__':

    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2=phi
    y2=np.sin(phi)+2
    x,y=contour_intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
