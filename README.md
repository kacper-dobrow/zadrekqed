# REST API for representativeness prediction

## Overview

The project implements an API service that predicts an object's representativeness. 
Representativeness is defined as the inverse of the mean distance from k nearest neighbours increased by 1.
The API accepts an array of n objects and then utilizes an ensemble of L models to predict their respective representativeness and return them in form of an n-long array.
The data used to train the ensemble was gathered at https://ssd.jpl.nasa.gov/tools/sbdb_query.html. It contains a table of Jupiter's Trojans with their orbital parameters. These were then used to obtain the asteroids position in a cartesian frame expressed in astronomical units. This choice of data allows for a literal interpretation of distance between objects and simplifies understanding of object representativeness.

## Project Structure

- `data_processing.py`: Contains functions for loading and splitting the dataset.
- `representativeness.py`: Contains the function for calculating representativeness.
- `model_training.py`: Contains functions for training the models with early stopping and saving them.
- `app.py`: RESTful API for making predictions using the trained models.
- `send_requests.py`: Script to send test requests to the API.
- `requirements.txt`: List of required Python packages.

## Installation

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
