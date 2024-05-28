from sklearn.metrics import pairwise_distances
import numpy as np

def calculate_representativeness(subset, K):
    distances = pairwise_distances(subset)
    nearest_distances = np.sort(distances, axis=1)[:, 1:K+1]
    avg_distances = np.mean(nearest_distances, axis=1)
    representativeness = 1 / (1 + avg_distances)
    return representativeness