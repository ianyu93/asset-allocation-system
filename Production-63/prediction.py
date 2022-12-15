################ Import Necessary Packages ################
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

################ Setup ################
train_data = sys.argv[1]
validation_data = sys.argv[2]
model_path = sys.argv[3]
pred_path = sys.argv[4]

# Read train data into a dataframe
train_df = pd.read_csv(train_data, index_col = 0)

# Set train_df index into datetime datatype
train_df.index = pd.to_datetime(train_df.index)

# Read the validation dataframe
val_df = pd.read_csv(validation_data, index_col = 0)

# Set val_df index into datetime datatype
val_df.index = pd.to_datetime(val_df.index)

# Load model for the specific asset to make prediction
model = keras.models.load_model(model_path)

################ Defining Functions to Create a Prediction Pipeline ################
def preprocessing(train_df, val_df, test_size, target, date_range):
    '''
    A funtion that preprocess the dataframe by:
        - Performing train-test split based on desired test size
        - Scaling the validation data with a MinMaxScaler
        - Reshaping the train set from 2D to 3D
        - Return just the features in 3D shape
    '''
    assert isinstance(train_df, pd.DataFrame), 'dataframe must be a Pandas DataFrame'
    assert isinstance(test_size, int), 'test_size must be an integer'
    assert isinstance(target, str), 'target must be a string'

    # Train-test split for the training dataset to keep the scaler exactly the same when training the model
    # Test set, however, is actually validation data
    train = train_df[:-test_size]
    test = val_df[:]

    # Setting features and target for train test
    X_train = train.drop(train_df.columns[0], axis = 1)

    # Setting features and target for validation data
    X_test = test.drop(train_df.columns[0], axis = 1)

    # Fitting MinMax Scaler onto the train set and convert into a DataFrame to keep the Timestamps
    mmscaler = MinMaxScaler()

    X_train_mms = pd.DataFrame(
        data = mmscaler.fit_transform(X_train),
        columns = X_train.columns,
        index = train.index
        )

    # Tranform the test set with mmscaler
    X_test_mms = pd.DataFrame(
        data = mmscaler.transform(X_test),
        columns = X_test.columns,
        index = date_range
        )

    return np.array(X_test_mms).reshape(
        np.array(X_test_mms).shape[0], -1, X_test.shape[1]
    )

def prediction(model, X, date_range, asset):
    '''
    A function that takes in a 3D shape array, pass through the model to evaluate, 
    reasign the index as the time the model is predicting, and return a dataframe of predicted values
    '''
    return pd.DataFrame(
        data=(model.predict(X).reshape(X.shape[0], -1)),
        index=date_range,
        columns=[f'{asset}'],
    )

################ Prediction ################
# Date range is starting from tomorrow, as the model predicts the next day of the last valid value
# How many days in the future is forecasted is based on how many rows for val_df
# Frequency is set to business days only, Monday to Friday
# Normalize to keep at daily timestamps only
date_range = pd.date_range(
    start = val_df.index[62], 
    periods = val_df.shape[0],
    freq = 'B',
    normalize = True
    )
# Create asset variable to obtain string of the asset name in order to create name stem in predicted column
asset = train_df.columns[0].replace(" Today","")

# Obtain an array of features to predict for the next number of days
# Target is the first column of train_df
# test_size is kept at 63, consistent with scaling process in hypertune and training 
X = preprocessing(
    train_df = train_df, 
    val_df = val_df, 
    test_size = 126, 
    target = train_df.columns[0], 
    date_range = date_range,
    )

# Create a dataframe of prediction for the next number of days based on feature values in X
pred = prediction(
    model = model, 
    X = X, 
    date_range = date_range,
    asset = asset
    )

# Save the result to a csv file
pred.to_csv(pred_path)