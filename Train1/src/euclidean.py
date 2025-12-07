import numpy as np


def euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    """

    diff = p1 - p2
    s = np.sum(diff**2)
    d = np.sqrt(s)
    return d


def euclidean_distance_matrix(points):
    """
    Calculate a matrix of Euclidean distances between points.
    """

    n = len(points)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            d = euclidean_distance(points[i], points[j])
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix
