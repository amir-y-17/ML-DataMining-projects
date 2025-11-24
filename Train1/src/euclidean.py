import math
import numpy as np


def euclidean_distance(p1, p2):
    """
    Calculate Euclidean distance between two points.
    """

    s = sum((p1[i] - p2[i]) ** 2 for i in range(len(p1)))
    d = math.sqrt(s)
    return d


def euclidean_distance_matrix(points):
    """
    Calculate a matrix of Euclidean distances between points.
    """

    n = len(points)
    matrix = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            d = euclidean_distance(points[i], points[j])
            matrix[i, j] = round(d, 3)
            matrix[j, i] = round(d, 3)

    return matrix
