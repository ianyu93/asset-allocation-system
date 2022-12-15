################ Import Necessary Packages ################
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Bidirectional,TimeDistributed, RepeatVector
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

################ Setup ################
train_data = sys.argv[1]
param_data = sys.argv[2]
save_model = sys.argv[3]

# Read the train dataset
df = pd.read_csv(train_data, index_col = 0)

# To ensure the index is set at datetime
df.index = pd.to_datetime(df.index)

################ Defining Functions to Create a Training Pipeline ################
def preprocessing(dataframe, test_size, target):
    '''
    A funtion that preprocess the dataframe by:
        - Performing train-test split
        - Scaling the data with a MinMaxScaler
        - Reshaping the train set from 2D to 3D
        - Return the features and the target in 3D
    '''
    assert isinstance(dataframe, pd.DataFrame), 'dataframe must be a Pandas DataFrame'
    assert isinstance(test_size, int), 'test_size must be an integer'
    assert isinstance(target, str), 'target must be a string'

    # Train-test Split, continously predicting for days of test_size
    train = dataframe[:-test_size]
    test = dataframe[-test_size:]

    # Setting features and target for train test
    X_train = train.drop(target, axis = 1)
    y_train = train[target]

    # Setting features and target for test set
    X_test = test.drop(target, axis = 1)
    y_test = test[target]

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
        index = test.index
        )

    ## Calling train set and validation set for nth split, reshape to 3D to fit into LSTM
    # Reshape logic for X is (total number of data points, length of array of the last dimension, features)
    # Reshape logic for y is (total number of data points, length of array of the last dimension, features) 
    Xtrain = np.array(X_train_mms).reshape(np.array(X_train_mms).shape[0],-1,X_train_mms.shape[1])
    ytrain = np.array(y_train).reshape(np.array(y_train).shape[0],-1,1)

    # Return 3D features and target for the train set
    return Xtrain, ytrain

def training_model(best_param, Xtrain, ytrain):
    '''
    A function that takes the values for parameters, 3D features, and 3D target,
    perform training on a defined architecture, and saves the best model based on lowest RMSE    
    '''
    # Convert best_param dataframe to dictionary with index 0 to take the first row of data
    param = best_param.to_dict(orient = 'records')[0]

    ## Learning rate schedule code is from Tensorflow documentation:
    ## https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=param['initial_learning_rate'],
        decay_steps=param['decay_steps'],
        decay_rate=0.9, staircase = True)


    ## Modeling with sequential
    model = keras.Sequential()

    ## Building Bidrectional LSTM RNN Encoder
    # Configuration from param
    # Input shape logic is (sequence, features)
    model.add(Bidirectional(LSTM(
        units = param['LSTM_units1'], 
        activation='relu',
        input_shape = (param['sequence'],Xtrain.shape[1]),
        return_sequences=False)))

    # Parameters for tuning the number of input sequence
    model.add(RepeatVector(1))

    ## Building Bidrectional LSTM RNN Decoder
    # Configuration from param
    # Return sequence to pass through TimeDistributedDense layer
    # Input shape logic is (sequence, features)
    model.add(Bidirectional(LSTM(
        units = param['LSTM_units1'], 
        activation='relu',
        input_shape = (param['sequence'],Xtrain.shape[1]),
        return_sequences=True)))

    # TimeDistributedDense Layer to keep input one timestamp at a time
    model.add(TimeDistributed(Dense(1)))

    ## Compiling model with Adam optimizer
    # Defining the compile structure
    # Learning rate schedule code is from Tensorflow documentation:
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
    model.compile(
        optimizer = keras.optimizers.Adam(
            clipnorm=param['clipnorm'], 
            learning_rate = lr_schedule
            ),
        loss = 'mse', 
        metrics = 'RootMeanSquaredError'
        )

    # Define early stopping, monitoring loss function, stops if not improving for five times in a row
    es = EarlyStopping(
        monitor='loss', 
        min_delta=1,
        patience = 5, 
        verbose=0, 
        mode='auto'
        )

    # Define model checkpoint to save the best model based on the lowest RMSE
    mc = ModelCheckpoint(
        filepath = save_model,
        monitor="root_mean_squared_error",
        verbose=2,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch"
        )

    return model.fit(
        Xtrain, ytrain, epochs=1000, verbose=2, callbacks=[es, mc]
    )

################ Training ################

# Obtain 3D features and target based on the train dataframe and scaled before the last 63 trading days
Xtrain, ytrain = preprocessing(
    dataframe = df, 
    test_size = 63, 
    target = df.columns[0]
    )

# Read best parameter of the model architecture for each asset
best_param = pd.read_csv(param_data, index_col = 0)

# Pass through best parameters, 3D features, and 3D target to the training_model function and save the model for each asset
history = training_model(
    best_param = best_param,
    Xtrain = Xtrain,
    ytrain = ytrain)