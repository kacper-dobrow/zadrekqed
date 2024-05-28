import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


def load_data():
    # TODO: Find a good dataset
    pass


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
