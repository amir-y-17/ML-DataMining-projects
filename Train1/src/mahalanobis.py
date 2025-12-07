import numpy as np
from src.covariance import covariance_matrix


def mahalanobis_distance(p1, p2, cov_matrix_inv):
    """
    Calculate Mahalanobis distance between two points given the covariance matrix.
    """

    diff_vector = p1 - p2
    T_diff_vector = np.transpose(diff_vector)

    distance = np.sqrt(T_diff_vector @ cov_matrix_inv @ diff_vector)
    return distance


def mahalanobis_distance_matrix(points):
    """
    Calculate a matrix of Mahalanobis distances between points.
    """

    n = len(points)
    matrix = np.zeros((n, n), dtype=float)

    cov_matrix_inv = np.linalg.inv(covariance_matrix(points))

    for i in range(n):
        for j in range(i, n):
            d = mahalanobis_distance(points[i], points[j], cov_matrix_inv)
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix
