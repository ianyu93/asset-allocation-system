################ Import Necessary Packages ################
import os
import sys
import numpy as np
import pandas as pd


#################### Setup ####################
values_from = f"{sys.argv[1]}"
target = f"{sys.argv[2]}"


# Read the clean dataset
df = pd.read_csv("data/cleaned_df.csv", index_col = 0)

# Ensure the index datatype is datetime rather than object
df.index = pd.to_datetime(df.index)

# Copy values from a column to set as the target for current dataset and predictor, inserting to index 0 to ensure target is always the first column
df.insert(
    loc = 0,
    column = target,
    value = df[values_from],
    allow_duplicates = True
    )

#################### Adding Features ####################
## Daily featuers
# SPX/DXY Relative Strength Line, increase means stock market is performing better
df['SPX/DXY RSL'] = df['SPX Close'] / df['DXY Close']

# SPX/WTI Relative Strength Line, increase means stock market is performing better
df['SPX/WTI RSL'] = df['SPX Close'] / df['WTI price']

# SPX/US10Y Relative Strength Line, increase means stock market is performing better
df['SPX/10YR RSL'] = df['SPX Close'] / df['10YR yields']

# SPX/GOLD Relative Strength Line, increase means stock market is performing better
df['SPX/GOLD RSL'] = df['SPX Close'] / df['GOLD price']

# GOLD/US10Y Relative Strength Line, increase means gold market is performing better
df['GOLD/10YR RSL'] = df['GOLD price'] / df['10YR yields']

## Yield Difference
# Features implying the yield curve
df['2Y/10Y yield difference'] = df['10YR yields'] - df['2YR yields']
df['3M/2Y yield difference'] = df['2YR yields'] - df['3MO yields']
df['5Y/10Y yield difference'] = df['10YR yields'] - df['5YR yields']
df['3M/10Y yield difference'] = df['10YR yields'] - df['3MO yields']
df['3M/5Y yield difference'] = df['3MO yields'] - df['5YR yields']

## Features for weekly moving average
# SPX, EMA 5 to capture momentum
df['SPX EMA5'] = df['SPX Close'].ewm(span=5).mean()

# SPX/DXY Relative Strength Line, 5-day simple moving average to capture general trend 
df['SPX/DXY RSL MA5'] = df['SPX/DXY RSL'].rolling(window=5).mean()

# SPX/WTI Relative Strength Line, 5-day simple moving average to capture general trend 
df['SPX/WTI RSL MA5'] = df['SPX/WTI RSL'].rolling(window=5).mean()

# WTI/GOLD Relative Strength Line, 5-day simple moving average to capture general trend 
df['WTI/GOLD RSL MA5'] = (df['WTI price'] / df['GOLD price']).rolling(window=5).mean()

# WTI/DXY Relative Strength Line, exponential moving average to capture momentum
df['WTI/DXY RSL EMA5'] = (df['WTI price'] / df['DXY Close']).ewm(span=5).mean()

## Features for monthly moving average
# SPX EMA13 for relatively short term momentum, capturing weekly momentum
df['SPX EMA13'] = df['SPX Close'].ewm(span=13).mean()

# SPX EMA 26 for relatively short term momentum, capturing monthly momentum
df['SPX EMA26'] = df['SPX Close'].ewm(span=26).mean()

# SPX/US10Y Relative Strength Line, 20-day simple moving average to capture general trend 
df['SPX/10YR RSL MA20'] = df['SPX/10YR RSL'].rolling(window=20).mean()

# SPX/GOLD Relative Strength Line, 20-day simple moving average to capture general trend 
df['SPX/GOLD RSL MA20'] = df['SPX/GOLD RSL'].rolling(window=20).mean()

# Applying Bollinger for monthly moving average
# Codes from: https://medium.com/@lhessani.sa/beat-the-market-using-algorithmic-trading-and-bollinger-band-python-application-9a59d0b34451
df['Bollinger 20 Middle Band'] = df['SPX Close'].rolling(window=20).mean()
df['Bollinger 20 Upper Band'] = df['Bollinger 20 Middle Band'] + 1.96*df['SPX Close'].rolling(window=20).std()
df['Bollinger 20 Lower Band'] = df['Bollinger 20 Middle Band'] - 1.96*df['SPX Close'].rolling(window=20).std()

## Features for quarterly moving average
# Applying quarterly moving average for SPX
df['SPX EMA50'] = df['SPX Close'].ewm(span=50).mean()
df['Bollinger 60 Middle Band'] = df['SPX Close'].rolling(window=60).mean()
df['Bollinger 60 Upper Band'] = df['Bollinger 60 Middle Band'] + 1.96*df['SPX Close'].rolling(window=60).std()
df['Bollinger 60 Lower Band'] = df['Bollinger 60 Middle Band'] - 1.96*df['SPX Close'].rolling(window=60).std()

# GOLD/US10Y Relative Strength Line, 60-day simple moving average to capture general trend 
df['GOLD/10YR RSL 60'] = df['GOLD/10YR RSL'].rolling(window=60).mean()

# GOLD/DXY Relative Strength Line, 60-day simple moving average to capture general trend 
df['GOLD/DXY RSL MA60'] = (df['GOLD price'] / df['DXY Close']).rolling(window=60).mean()

## EMA on RSL
# SPX/DXY RSL
df['SPX/DXY RSL EMA 200'] = df['SPX/DXY RSL'].ewm(span = 200).mean()

# SPX/WTI RSL
df['SPX/WTI RSL EMA 200'] = df['SPX/WTI RSL'].ewm(span = 200).mean()

# SPX/US10Y RSL
df['SPX/10YR RSL EMA 200'] = df['SPX/10YR RSL'].ewm(span = 200).mean()

# SPX/GOLD RSL
df['SPX/GOLD RSL EMA 200'] = df['SPX/GOLD RSL'].ewm(span = 200).mean()

# GOLD/US10Y RSL
df['GOLD/10YR RSL EMA 200'] = df['GOLD/10YR RSL'].ewm(span = 200).mean()


#################### Transforming to Structured Learning Problem ####################
# Take an list of columns except target and CPI Annual Rate to shift
column_list = list(df.drop([target,'CPI annual rate'], axis = 1).columns)

# Creating two copies, training data for training the model, validation data for final prediction
training = df.copy()
validation_data = df.copy()

## Shifting values back at least 21 trading days 
# For every column in the column list
for c in column_list:
    for n in (63,126,252):      
        # Shift the data back 21, 42, and 63 trading days, and name the new column with value name and n day lag
        training[f'{c} {n} Day Lag'] = training[f'{c}'].shift(n)
    # Dropping the original columns so that we are left with lagged data except the target
    training.drop(f'{c}', axis = 1, inplace = True)

## Shifting values back with the same interval as training data, to predict the future with the latest data
# For every column in the column list
for c in column_list:
    for n in (0, 63, 126):      
        # Shift the data back 0, 21, and 42 trading days, and name the new column with value name and n day lag
        validation_data[f'{c} {n} Day Lag'] = validation_data[f'{c}'].shift(n)
    # Dropping the original columns so that we do not have the same day value left
    validation_data.drop(f'{c}', axis = 1, inplace = True)

# Dropping newly introduced missing values as  a result of shifting
training.dropna(axis = 0, how = 'any', inplace = True)
validation_data.dropna(axis = 0, how = 'any', inplace = True)

# Export the engineered dataframe, 63 trading days in a quarter
training.to_csv(f'train/{sys.argv[3]}_training_data.csv')
validation_data[-126:].to_csv(f'validate/{sys.argv[3]}_validation_data.csv')