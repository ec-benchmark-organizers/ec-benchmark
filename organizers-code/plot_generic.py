"""
Generic plotting methods.

This file was copied from: https://raw.githubusercontent.com/virocon-
organization/viroconweb/master/contour/plot_generic.py
"""
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
import shapely.geometry as geometry
import numpy as np
import math
from shapely.geometry import MultiPoint
from shapely.geometry import Point
import warnings

# Based on: http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python
def alpha_shape(points, alpha):
    """
    Computes the alpha shape (concave hull) of a set of points

    Parameters
    ----------
    points : MultiPoint,
        Iterable container of points.
    alpha : float,
        Alpha value to influence the gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers. Too large and you lose
        everything!

    Returns
    -------

    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
    coords = np.array([point.coords[0]
                       for point in points])
    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points

def convert_ndarray_list_to_multipoint(ndarray_list):
    """
    Converts an array list to a MultiPoint, which is required by alpha_shape

    Parameters
    ----------
    ndarray_list : list of ndarray,
        A list of data points.
        # TODO: Type "list of ndarray" is inconsistent with the documentation
        of plot.plot_contour which claims it is a list of float. Check that.

    Returns
    -------
    points: MultiPoint,
        The data points as an MultiPoint object such that the method
        alpha_shape can work with it.
    """
    points_list = []
    data_dimension = len(ndarray_list)
    for i in range(len(ndarray_list[0])):
        if data_dimension == 2:
            points_list.append(Point(ndarray_list[0][i], ndarray_list[1][i]))
        elif data_dimension == 3:
            points_list.append(Point(ndarray_list[0][i], ndarray_list[1][i], ndarray_list[2][i]))
        else:
            warnings.warn("4-Dim plot or higher is not supported", DeprecationWarning)
    points = MultiPoint(points_list)
    return points
