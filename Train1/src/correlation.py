import numpy as np


def compute_means(data: np.ndarray) -> np.ndarray:
    """
    Compute mean of each feature manually.
    """
    n, d = data.shape
    means = np.zeros(d, dtype=float)

    for j in range(d):
        s = 0.0
        for i in range(n):
            s += data[i, j]
        means[j] = s / n

    return means


def compute_std(data: np.ndarray, means: np.ndarray) -> np.ndarray:
    """
    Compute standard deviation of each feature manually.
    (Using sample formula: sqrt( sum((x-mean)^2) / (n-1) ))
    """
    n, d = data.shape
    stds = np.zeros(d, dtype=float)

    for j in range(d):
        s = 0.0
        for i in range(n):
            diff = data[i, j] - means[j]
            s += diff * diff
        stds[j] = np.sqrt(s / (n - 1))

    return stds


def correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix manually using Pearson correlation.
    """
    n, d = data.shape
    corr = np.zeros((d, d), dtype=float)

    # Step 1: means
    means = compute_means(data)

    # Step 2: std deviations
    stds = compute_std(data, means)

    # Step 3: correlation
    for j in range(d):
        for k in range(d):

            numerator = 0.0
            for i in range(n):
                numerator += (data[i, j] - means[j]) * (data[i, k] - means[k])

            denominator = (n - 1) * stds[j] * stds[k]

            corr[j, k] = numerator / denominator

    return corr
