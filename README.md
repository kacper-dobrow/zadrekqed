# REST API for representativeness prediction

## Overview

The project implements an API service that predicts an object's representativeness. 
Representativeness is defined as the inverse of the mean distance from k nearest neighbours increased by 1.
The API accepts an array of n three dimensional position points and then utilizes an ensemble of L models to predict their respective representativeness scores and return them in form of an n-long array.
The data used to train the ensemble was gathered at [NASA Small-Body Database Query](https://ssd.jpl.nasa.gov/tools/sbdb_query.html). It contains a table of Jupiter's Trojans with their orbital parameters. These are then used to obtain the asteroids position in a cartesian frame expressed in astronomical units. This choice of data allows for a literal interpretation of distance between objects and simplifies understanding of object representativeness.

## Project Structure

- `data_processing.py`: Contains functions for loading and splitting the dataset.
- `representativeness.py`: Contains the function for calculating representativeness.
- `model_training.py`: Contains functions for training the models and saving them.
- `app.py`: RESTful API for making predictions using the trained models.
- `send_requests.py`: Script to send test requests to the API.
- `requirements.txt`: List of required Python packages.

## Installation

Python 3.12 is recommended. However, everything should run with Python 3.8.
Make sure you install all the required packages
```bash
pip install -r requirements.txt
```
## Usage

### Training the Models

Run the model_training.py script to train the models and save them:
```bash
python model_training.py
```
The trained models will be saved to models.pkl.

### Running the API

Start the Flask API server:

```bash
python app.py
```
The API will be available at http://127.0.0.1:5000/predict.

### Sending Requests

Use the send_requests.py script to send test requests to the API:

```bash
python send_requests.py
```
This script sends both valid and invalid requests to the API and prints the responses.

## Issues and further steps

The ensemble model that is used in this project utilizes the `sklearn.tree.DecisionTreeRegressor` models. The training dataset is split into L subsets, of which each is used as a training set for a Decision Tree model. The models are obviously overfitted to their respective subsets and are bad for predictions on the whole dataset. Measures preventing overfitting need to be introduced.

The app operation is verified with only three unit tests. More extensive tests need to be implemented.
