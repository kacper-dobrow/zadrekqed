from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
# models = joblib.load('models.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame(input_data)
    # predictions = np.mean([model.predict(input_df) for model in models], axis=0)
    predictions = {"Hello": "World"}
    # return jsonify(predictions.tolist())
    return predictions


if __name__ == '__main__':
    app.run(debug=True)
