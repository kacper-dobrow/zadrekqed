from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
models = joblib.load('models.pkl')


def validate_input(input_data):
    """
    Validate that the input data is a list of 3D vectors.

    Args:
        input_data (list): The input data to validate.

    Returns:
        bool: True if the input data is valid, False otherwise.
    """
    if not isinstance(input_data, list):
        return False

    for item in input_data:
        if not isinstance(item, list):
            return False
        if len(item) != 3:
            return False
        if not all(isinstance(value, (int, float)) for value in item):
            return False

    return True


@app.route('/predict', methods=['POST'])
def predict_representativeness():
    input_data = request.json
    if not validate_input(input_data):
        return jsonify({"error": "Bad Request: Input data should be a list of 3D vectors."}), 400
    input_data = request.json
    input_df = pd.DataFrame(input_data)
    predictions = np.mean([model.predict(input_df) for model in models], axis=0)
    return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(debug=True)
