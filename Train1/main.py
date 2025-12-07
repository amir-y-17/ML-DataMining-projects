from src import *
import numpy as np

points = load_data()

# ========= Euclidean Distance Matrix =========
euclidean_matrix = euclidean_distance_matrix(points)
print(euclidean_matrix)

# ========= Mahalanobis Distance Matrix =========
mahalanobis_matrix = mahalanobis_distance_matrix(points)
print(mahalanobis_matrix)
