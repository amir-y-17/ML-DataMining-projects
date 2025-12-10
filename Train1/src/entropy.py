import numpy as np


def entropy_of_feature(values, num_bins=10):
    """
    Compute entropy of a single feature (1D array) manually.
    Data is continuous → so we discretize into bins manually.
    """

    n = len(values)

    # Step 1: Find min/max
    v_min = np.min(values)
    v_max = np.max(values)

    # Prevent division by zero if all values are same
    if v_min == v_max:
        return 0.0

    # Step 2: Create bins manually
    bin_size = (v_max - v_min) / num_bins

    # Step 3: Count values in each bin
    counts = [0] * num_bins

    for x in values:
        idx = int((x - v_min) / bin_size)
        if idx == num_bins:
            idx -= 1
        counts[idx] += 1

    # Step 4: Convert counts to probabilities
    probs = [c / n for c in counts if c > 0]

    # Step 5: Compute entropy manually
    entropy = 0.0
    for p in probs:
        entropy += -p * np.log2(p)

    return entropy


def entropy_all_features(data, num_bins=10):
    """
    Compute entropy for each feature of the dataset manually.
    """
    n, d = data.shape
    entropies = np.zeros(d, dtype=float)

    for j in range(d):
        entropies[j] = entropy_of_feature(data[:, j], num_bins=num_bins)

    return entropies
