import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

def load_data():
    """
    Load the Iris dataset.

    The Iris dataset is a classic dataset for classification tasks. It contains 150 samples
    of iris flowers, each described by four features: sepal length, sepal width, petal length,
    and petal width.

    Returns:
        pd.DataFrame: A DataFrame containing the Iris dataset with columns for each feature.
    """
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df

def split_data(data, L):
    """
    Split the data into L roughly equal parts randomly.

    Args:
        data (pd.DataFrame): The dataset to split.
        L (int): The number of parts to split the data into.

    Returns:
        list of pd.DataFrame: A list of DataFrames, each containing a roughly equal part of the data.
    """
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data
    return np.array_split(data, L)

