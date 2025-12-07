import numpy as np


def covariance_matrix(data):
    """
    Calculate the covariance matrix for the given dataset.
    """
    n = len(data)
    d = len(data[0])

    means = [0.0] * d

    for j in range(d):
        s = 0
        for i in range(n):
            s += data[i][j]
        means[j] = s / n

    # Center the data
    data_centered = []
    for i in range(n):
        centered_row = []
        for j in range(d):
            centered_row.append(data[i][j] - means[j])
        data_centered.append(centered_row)

    # Calculate covariance matrix
    S = np.zeros((d, d), dtype=float)

    for j in range(d):
        for k in range(d):
            s = 0.0
            for i in range(n):
                s += data_centered[i][j] * data_centered[i][k]
            S[j, k] = s / (n - 1)

    return S
