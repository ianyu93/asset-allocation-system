#################### Import necessary packages ####################
import os
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
import kerastuner as kt
import IPython

################ Setup ################
train_data = sys.argv[1]
param_path = sys.argv[2]

# Read the train dataset
df = pd.read_csv(train_data, index_col = 0)

# To ensure the index is set at datetime
df.index = pd.to_datetime(df.index)

################ Defining Class and Methods to Create a Hypertuning Pipeline ################
# Define a callback to clear the training outputs at the end of every training step.
# Code from https://www.tensorflow.org/tutorials/keras/keras_tuner
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)
# Early stopping to prevent overfitting
er = EarlyStopping(
    monitor='val_root_mean_squared_error',
    patience=5
    )


class HyperTuningModel():

    '''
    A class that takes in a dataframe and creates objects for different steps, including:
        - preprocessing to perform train-test split and scale the data,
        - modeul_builder to define the model architecture hypertuning parameters
        - val_size to determine the appropriate number of splits for TimeSeriesSplit
        - LSTM_tuner to perform hypertuning search
        - best_param_table to save all parameters and scores for each split for future reference
    '''
    # Initializing the function, which takes in a dataframe
    def __init__(self, dataframe):
        assert isinstance(dataframe, pd.DataFrame), 'Must pass a Pandas DataFrame'
        self.dataframe = dataframe

    def preprocessing(self, test_size, target):
        '''
        A funtion that preprocess the dataframe by:
            - Performing train-test split
            - Scaling the data with a MinMaxScaler
            - Return train, test, X_train_mms, X_test_mms, y_train, y_test
        '''
        assert isinstance(self.dataframe, pd.DataFrame), 'dataframe must be a Pandas DataFrame'
        assert isinstance(test_size, int), 'test_size must be an integer'
        assert isinstance(target, str), 'target must be a string'
        
        # Train-test Split, continously predicting for days of test_size
        train = self.dataframe[:-test_size]
        test = self.dataframe[-test_size:]
        
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
            index = test.index)
        
        # Return train set, test set, scaled X's, and y's for both sets
        return train, test, X_train_mms, X_test_mms, y_train, y_test
    
    ## Code modified from Keras Tuner: https://www.tensorflow.org/tutorials/keras/keras_tuner 

    def model_builder(self, hp):
        '''
        Define a model architecture as well as the parameters to hypertune
        '''
        model = keras.Sequential()

        ####################
        # Model Parameters #
        ####################
        ## Bidirectional LSTM Layer, tune LSTM_units and sequence for input_shape
        # Parameters for tunining the number of units
        LSTM_units1 = hp.Int(
            'LSTM_units1', 
            min_value = 8, 
            max_value = 256, 
            step = 8
            )
        
        # Parameters for tuning the number of input sequence
        sequence = hp.Int(
            'sequence', 
            min_value = 1, 
            max_value = 200, 
            step = 1
            )

        #######################
        # Model Archietecture #
        #######################

        # Defining the encoder, input_shape is (sequence, features)
        model.add(Bidirectional(LSTM(
            units = LSTM_units1, 
            activation='relu',
            input_shape = (sequence,X_train_mms.shape[1]),
            return_sequences=False
            )))

        # Repeat Vector as a bridge between encoder and decoder
        model.add(RepeatVector(1))

        # Defining the decoder, input_shape is (sequence, features)
        # Return sequence to pass through TimeDistributedDense layer
        model.add(Bidirectional(LSTM(
            units = LSTM_units1, 
            activation='relu',
            input_shape = (sequence,X_train_mms.shape[1]),
            return_sequences=True
            )))

        # TimeDistributedDense Layer to keep input one timestamp at a time
        model.add(TimeDistributed(Dense(1)))

        ## Compiling with Adam optimizer
        # Parameters for tuning the initial learning rate
        hp_learning_rate = hp.Choice(
            'initial_learning_rate', 
            values = [1e-2, 1e-3, 1e-4]
            ) 

        # Parameters for tuning decay steps to decay the learning rate over number of observations
        hp_decay_steps = hp.Choice(
            'decay_steps', 
            values = [1e3, 1e4]
            )

        # Parameters for tuning clip norm to perform graidient clipping 
        hp_clipnorm = hp.Choice(
            'clipnorm', 
            values = [0.1,0.5,1.0]
            )
        
        # Defining the compile structure
        # Learning rate schedule code is from Tensorflow documentation:
        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
        # Loss function with Mean Squared Error to find the best fit / minimize deviation from the true values
        # Root Mean Squared Error as a more human readable metrics to evaluate the model
        model.compile(
            optimizer = keras.optimizers.Adam(
                learning_rate = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=hp_learning_rate,
                    decay_steps=hp_decay_steps,
                    decay_rate=0.9,
                    staircase = True
                    ),
                clipnorm=hp_clipnorm),
            loss = 'mse',
            metrics = ['RootMeanSquaredError']
            )

        # Returning the model architecture
        return model

    def val_size(self, preprocessed_data, size):
        '''
        A function that passes through the scaled dataset, perform TimeSeriesSplit, 
        and find the optimal number of splits given the desired validation size
        '''
        assert isinstance(preprocessed_data,tuple), 'preprocessed_data must be a tuple'
        assert isinstance(size, int), 'size must be an integer'
        # Minimum number of splits is 2
        n = 2

        # Validation size is the size of the validation set given the number of split
        val_size = preprocessed_data[2].shape[0]

        # While validation set is larger than desired size, keep increasing the number of splits
        while val_size > size:
            # Defining time series splits
            tscv = TimeSeriesSplit(n_splits = n)
            
            ## Turn tscv into a DataFrame to work with the split time series
            # Returns two columns, column 0 is train indexes, column 1 is validation indexes
            # Each row is a split, 4th row would be the 4th split, for example
            timestep = pd.DataFrame(tscv.split(preprocessed_data[2]))

            # Increase the number of splits for next loop
            n += 1

            # Validation size is the length of the timestamps returned through TimeSeriesSplit
            val_size = len(timestep[1][0])

            # Number of splits performed in the current loop
            splits = len(timestep[1])
        
        # Return the number of splits
        return splits
    
    def LSTM_tuner(self, splits, start_split, model_builder, name):
        '''
        Define a function that takes in X_train and y_train and perform time series split.
        The function would also take the argument of number of splits as well as which split to start hypertune against.
        The method would also include the model_builder to hypertune.
        The last argument 'name' is to decide the project name
        '''
        
        
        ## Time Series Split
        # Defining time series splits
        tscv = TimeSeriesSplit(n_splits = splits)

        ## Turn tscv into a DataFrame to work with the split time series
        # Returns two columns, column 0 is train indexes, column 1 is validation indexes
        # Each row is a split, 4th row would be the 4th split, for example
        timestep = pd.DataFrame(tscv.split(X_train_mms))

        ## Create number of steps based on number of splits to hypertune on the last few splits onlny
        # Range of numbers, the first number is the split to start with, as there is no point to start with the very first split
        # len(timestep) would return the number of splits, in effect returning the last split
        n_step = list(range(start_split,len(timestep),1))

        ## Create lists to get models, parameters, and metrics score for each split
        model_list = []
        param_list = []
        score_list = []

        ## Create for loop to validate each split into the model, for each step: 
        for n in n_step:
            ## Calling train set and validation set for nth split, reshape to 3D to fit into LSTM
            # Reshape logic for X is (total number of data points, length of array of the last dimension, features) 
            Xtrain = np.array(X_train_mms.iloc[timestep[0][n]]).reshape(np.array(X_train_mms.iloc[timestep[0][n]]).shape[0],-1,X_train_mms.shape[1])
            Xvalidation = np.array(X_train_mms.iloc[timestep[1][n]]).reshape(np.array(X_train_mms.iloc[timestep[1][n]]).shape[0],-1,X_train_mms.shape[1])
            
            # Reshape logic for y is (total number of data points, length of array of the last dimension, features)
            ytrain = np.array(y_train.iloc[timestep[0][n]]).reshape(np.array(y_train.iloc[timestep[0][n]]).shape[0],-1,1)
            yvalidation = np.array(y_train.iloc[timestep[1][n]]).reshape(np.array(y_train.iloc[timestep[1][n]]).shape[0],-1,1)

            ## Define tuner to use model_builder, using RMSE for validation as objective to tune
            # Bayesian Optimization utilizes regression-like math to find the best parameter, more thorough but heavier than Random Search
            # Finding the best result for Root Mean Squared Error for the validation set as the objective
            # Maximum number of trials is 10
            # Project name logic "{name of the target}_trial-{nth split}" 

            os.mkdir(f"trials/{name}_{n}/")

            tuner = kt.BayesianOptimization(
                model_builder,
                kt.Objective("val_root_mean_squared_error", direction="min"),
                max_trials = 3,
                directory=os.path.normpath('C:/'),
                project_name = f"trials/{name}_{n}/0"
                    )

            # Set tuner to search for the best parameter for the given 3-D train and validation set
            tuner.search(
                Xtrain, ytrain, 
                epochs = 1000, 
                validation_data = (Xvalidation, yvalidation), 
                verbose = 2, 
                callbacks = [ClearTrainingOutput(), er]
                )  
            
            # Collect a list of best models
            best_models = tuner.get_best_models(num_models = 1)

            # Collect a list of best parameters in each split
            best_hps = tuner.get_best_hyperparameters(num_trials = 1)

            # For each loop, append best model to model_list, best param to param_list
            model_list.append(best_models)
            param_list.append(best_hps)

            # For each loop, evaluate the validation set with the best model, and append score to score_list 
            score_list.append(best_models[0].evaluate(Xvalidation, yvalidation)[1])

        # Return the index of the best parameters in the score_list based on the minimum value of RMSE
        best_param = param_list[score_list.index(min(score_list))]

        # Take the index from previous code, return the hypertuned values for each parameter, and convert to a dataframe
        best_param = pd.json_normalize(best_param[0].values)

        # The final output is a dataframe of best parameters, a list of models, a list of parameters, and a list of scores
        return best_param, model_list, param_list, score_list
    
    def best_param_table(self, param_list, score_list):
        '''
        A function that creates a pandas table of best params and score in each validation.
        Although in the previous method we only get the best parameter of all validation sets,
        we keep all the parameters and their evaluation score for future analysis on the hypertuning result
        '''
        
        # Paramters data is the values of parameters from the best trial for each split in the param_list
        param_data = [x[0].values for x in param_list]

        # Convert to a dataframe that has parameters of each split, each parameter is a feature
        table = pd.DataFrame(data = param_data)

        # Append the evaluation score to the dataframe
        table['score'] = score_list

        # Return a table with parameters and an evaluation score as the final dataframe
        return table

# Istantiate HyperTuningModel and pass through the train dataframe
htm = HyperTuningModel(df)

# Preprocess the train dataframe, target is 'asset Today'
processed = htm.preprocessing(
    test_size = 63, 
    target = df.columns[0])

# Determine the optimal number of splits based on keeping a validation size of one year of trading days
split_num = htm.val_size(
    preprocessed_data = processed, 
    size = 252
    )

# 'processed' is a tuple, and obtain train set, test set, scaled X's and y's for both sets
train, test, X_train_mms, X_test_mms, y_train, y_test = processed

# Perform tuner search, over last 5 split, trial name is based on target name
best_param, model_list, param_list, score_list = htm.LSTM_tuner(
    splits = split_num, 
    start_split = split_num-5, 
    model_builder = htm.model_builder, 
    name = f"{df.columns[0]}"
    )

# Obtain the best_param_table by passing through param_list and score_list from previous code
best_param_table = htm.best_param_table(
    param_list = param_list, 
    score_list = score_list
    )

# Obtaining just the asset name to save a csv file for best_param_table
project = df.columns[0].replace(" Today","")

# Export both just the best parameters for training purposes and best_param_table for manual analysis
best_param.to_csv(param_path)
best_param_table.to_csv(f"param/{project}_best_param.csv")