import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    # loaded data is the orbital parameters of Jupiter's Trojans obtained from https://ssd.jpl.nasa.gov/tools/sbdb_query.html
    data = pd.read_csv("orbital_parameters.csv")
    return data


def calculate_positions(df):
    """ Calculates the cartesian coordinates of celestial bodies given their orbital parameters """
    # Conversion from radians to degrees
    df['i_rad'] = np.radians(df['i'])
    df['om_rad'] = np.radians(df['om'])
    df['w_rad'] = np.radians(df['w'])
    # as the orbits are nearly circular I will assume that
    # mean anomaly equals true anomaly
    # and the distance between the Sun and the asteroid equals the semimajor axis
    # for calculation simplicity sake
    df['ma_rad'] = np.radians(df['ma'])

    # Radius
    # df['r'] = df['a'] * (1 - df['e']**2) / (1 + df['e'] * np.cos(df['ma_rad']))
    df['r'] = df['a']

    # Calculation of the cartesian coordinates
    df['x'] = df['r'] * (np.cos(df['om_rad']) * np.cos(df['w_rad'] + df['ma_rad']) - np.sin(df['om_rad']) * np.sin(df['w_rad'] + df['ma_rad']) * np.cos(df['i_rad']))
    df['y'] = df['r'] * (np.sin(df['om_rad']) * np.cos(df['w_rad'] + df['ma_rad']) + np.cos(df['om_rad']) * np.sin(df['w_rad'] + df['ma_rad']) * np.cos(df['i_rad']))
    df['z'] = df['r'] * np.sin(df['i_rad']) * np.sin(df['w_rad'] + df['ma_rad'])

    return df[['x', 'y', 'z']]


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


if __name__=="__main__":
    df = load_data()
    positions = calculate_positions(df)
    fig, ax = plt.subplots()
    ax.scatter(positions["x"], positions["y"])
    plt.show()