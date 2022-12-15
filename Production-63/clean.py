################ Import Necessary Packages ################
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import time
import yfinance as yf

from fredapi import Fred
import quandl

################ APIs ################
# Provide API Keys to FRED and Quandl
fred = Fred(api_key="3e7ce9d3322d45b49f624720abd0f36a")
quandl.ApiConfig.api_key = "_gTGp-_JJ9kKR7-hCGT5"

## Request stock, bonds, dollar index, gold, WTI oil, and CPI from all APIs
stock = yf.Ticker("^GSPC").history(period="max")
yields = quandl.get("USTREASURY/YIELD")
usd = yf.Ticker("DX-Y.NYB").history(period="max")
gold = quandl.get("LBMA/GOLD")

wti = pd.DataFrame(fred.get_series_latest_release(
    'DCOILWTICO'), columns=["price"])
cpi = pd.DataFrame(fred.get_series_latest_release(
    'CPALTT01USA659N'), columns=["annual rate"])
time.sleep(1)


################ Imputation, Dropping Features, Data Wrangling ################
# Stock
# Dropping dividends and stock splits for stock df
stock = stock.drop(['Dividends', 'Stock Splits'], axis=1)

# Dollar index
# Dropping volume, dividends, and stock splits for the dollar idnex
usd = usd.drop(['Dividends', 'Stock Splits', 'Volume'], axis=1)

# WTI oil
# forward filling the missing values from statutory holidays landing on weekdays
wti = wti.fillna(method='ffill')

# Treasury yields
# yields drop columns with too many missing values
yields = yields.drop(['1 MO', '2 MO', '20 YR', '30 YR'], axis=1)

# yields forward fill missing values at random for '3 MO'
yields['3 MO'] = yields['3 MO'].fillna(method='ffill')
yields.columns = yields.columns.str.replace(" ","")

# Gold
# Keep only 'USD (AM)' for one daily values only and rename 'USD (AM)' to 'price'
gold = gold[['USD (AM)']].fillna(method='ffill').rename(columns={'USD (AM)': 'price'})

# Consumer Price Index
# CPI drop the first four years as they are all NaN values and set 2020 annual inflation to 1.1
cpi = cpi.dropna()
cpi2 = pd.DataFrame(index=['2020-01-01'], columns=['annual rate'], data=[1.1])
cpi2.index = pd.to_datetime(cpi2.index)
cpi = cpi.append(cpi2)


################ Putting All Dataframes Together ################
## Reindexing
# Creating a date range of our interest, 20 years worth of data is kept, in frequency of business day, in daily values format
date_range = pd.date_range(
    start=datetime.now() - timedelta(days=7300),
    end=datetime.now(),
    freq='B',
    normalize=True,
)

# Reindexing with forward fill to fill new missing values with last valid observation
stock = stock.reindex(index=date_range, method='ffill')
usd = usd.reindex(index=date_range, method='ffill')
yields = yields.reindex(index=date_range, method='ffill')
wti = wti.reindex(index=date_range, method='ffill')
gold = gold.reindex(index=date_range, method='ffill')
cpi = cpi.reindex(index=date_range, method='ffill')

## Concatenation of all dataframes into one single dataframe
# Add a string on the column names to indicate the market for each dataframe
stock.columns = "SPX " + stock.columns
yields.columns = yields.columns + ' yields'
usd.columns = "DXY " + usd.columns
wti.columns = "WTI " + wti.columns
gold.columns = "GOLD " + gold.columns
cpi.columns = "CPI " + cpi.columns

# Concatenating all dataframes
all_df = pd.concat([stock, yields, usd, gold, wti, cpi], axis=1)

# Exporting the dataframe as csv
all_df.to_csv('data/cleaned_df.csv')
