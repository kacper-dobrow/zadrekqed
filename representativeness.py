from sklearn.metrics import pairwise_distances
import numpy as np

def calculate_representativeness(subset, K):
    """
    Calculate the representativeness of each object in the subset.

    Representativeness is a measure of how well an object represents other objects in the dataset.
    Here it is defined as the inverse of the sum of one and the mean distance to the K nearest neighbors.

    Args:
        subset (np.ndarray): A subset of the data. Each row represents an object, and each column represents a feature.
        K (int): The number of nearest neighbors to consider.

    Returns:
        np.ndarray: An array of representativeness scores for each object in the subset.
    """
    distances = pairwise_distances(subset)
    nearest_distances = np.sort(distances, axis=1)[:, 1:K+1]
    avg_distances = np.mean(nearest_distances, axis=1)
    representativeness = 1 / (1 + avg_distances)
    return representativeness