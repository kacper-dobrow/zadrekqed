from sklearn.tree import DecisionTreeRegressor
import joblib
from data_processing import load_data, split_data, calculate_positions
from representativeness import calculate_representativeness


def train_model(data, L, K):
    """
    Train multiple models on the data split into L parts, using K nearest neighbors to calculate representativeness.

    Args:
        data (pd.DataFrame): The dataset to train the models on.
        L (int): The number of parts to split the data into.
        K (int): The number of nearest neighbors to consider when calculating representativeness.

    Returns:
        list of DecisionTreeRegressor: A list of trained models.

    Example:
        df = load_data()
        models = train_model(df, L=5, K=5)
    """
    splits = split_data(data, L)
    models = []
    all_reps = calculate_representativeness(data, K)
    for split in splits:
        reps = calculate_representativeness(split, K)
        model = DecisionTreeRegressor()
        model.fit(split, reps)
        # The below lines check the score of the models
        # The models are obviously overfitted to their own split and are bad for others
        # score1 = model.score(data, all_reps)
        # score2 = model.score(split, reps)
        # print(f"Score on the split: {score2}\nScore on the whole dataset: {score1}")
        models.append(model)
    return models

def save_models(models, filename):
    """
    Save the trained models to a file.

    Args:
        models (list of RandomForestRegressor): The list of trained models to save.
        filename (str): The name of the file to save the models to.

    Example:
        save_models(models, 'models.pkl')
    """
    joblib.dump(models, filename)

if __name__ == '__main__':
    df = calculate_positions(load_data())
    L = 5  # Number of) parts to split the data into
    K = 5  # Number of nearest neighbors to consider
    models = train_model(df, L, K)
    save_models(models, 'models.pkl')
