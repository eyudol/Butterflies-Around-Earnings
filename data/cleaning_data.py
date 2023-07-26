"""
This script reads data that was provided by an options data vendor in CSV format,
cleans it and re-performs some calculations in a more accurate manner,
and finally saves it to a parquet format with appropriate data types for
efficient reading and memory use later on in the project. I would have preferred to
use an even faster file format like feather but Dask annoying does not have support
for anything other than CSV and parquet.
"""

import pandas as pd
import numpy as np
import dask.dataframe as dd
from collections import defaultdict
import py_vollib_vectorized

# directories that contain data from vendor and the directory to which to write the cleaned data
DATA_DIR_TEMPLATE: str = "/Volumes/Seagate/Selected_Option_Data/SelectedOptionData/{} Option Data For 2018-2022.csv"
OUTPUT_DIR: str = "/Volumes/Seagate/Selected_Option_Data/CleanedParquets"

# get a list of paths with relevant top 250 tickers
tickers: list[str] = pd.read_excel('Top 250 Companies.xlsx', usecols=['Ticker'], dtype={'Ticker': 'string'})[
    'Ticker'].to_list()
paths: list[str] = [DATA_DIR_TEMPLATE.format(ticker) for ticker in tickers]

# read all files into dask dataframe with appropriate data types
dtype_dict: defaultdict = defaultdict(lambda: 'float32')
not_float_or_date: dict[str, str] = {'Symbol': 'string', 'PutCall': 'category', 'AskSize': 'int32', 'BidSize': 'int32'}
cols_to_use: list[str] = ['Symbol', 'ExpirationDate', 'AskPrice', 'AskSize', 'BidPrice', 'BidSize', 'PutCall',
                          'StrikePrice', 'UnderlyingPrice', 'DataDate']
dtype_dict.update(not_float_or_date)
date_cols: list[str] = ['ExpirationDate', 'DataDate']
ddf: dd = dd.read_csv(paths, blocksize=None, usecols=cols_to_use, parse_dates=date_cols, dtype=dtype_dict)

# add some columns that will be useful later
ddf['DTE']: dd.Series = ((ddf['ExpirationDate'] - ddf['DataDate']) / np.timedelta64(1, 'D')).astype('int32')
ddf['Spread']: dd.Series = (ddf['AskPrice'] - ddf['BidPrice']).astype('float32')
ddf['MidPrice']: dd.Series = ((ddf['AskPrice'] + ddf['BidPrice']) / 2).astype('float32')
ddf['StrikeProp']: dd.Series = (ddf['StrikePrice'] / ddf['UnderlyingPrice']).astype('float32')

# all values in the columns in this greater_than_0 list should be greater than or equal to zero
greater_than_0: list[str] = ['AskPrice', 'AskSize', 'BidPrice', 'BidSize', 'StrikePrice', 'UnderlyingPrice', 'DTE',
                             'Spread']
ddf: dd.Series = ddf[(ddf[greater_than_0] >= 0).all(axis=1)]


# Annoyingly, have to apply the Black-Scholes calculations separately to each partition of the dask dataframe because
# py_vollib does not support dask dataframes.
# ---------------------------- Functions for Black-Scholes calculations --------------------------------- #

# calculate implied volatility
def get_iv(df: pd.DataFrame) -> None:
    df['ImpliedVolatility']: pd.Series = py_vollib_vectorized.vectorized_implied_volatility(
        price=df['MidPrice'], S=df['UnderlyingPrice'], K=df['StrikePrice'], t=df['DTE'] / 365, r=.02,
        flag=df['PutCall'].str[:1], q=0, model='black_scholes_merton', return_as='numpy'
    )


# calculate option delta
def get_delta(df: pd.DataFrame) -> None:
    df['ImpliedVolatility']: pd.Series = py_vollib_vectorized.vectorized_delta(
        S=df['UnderlyingPrice'], K=df['StrikePrice'], t=df['DTE'] / 365, r=.02,
        flag=df['PutCall'].str[:1], sigma=df['ImpliedVolatility'], q=0, model='black_scholes_merton',
        return_as='numpy'
    )


# calculate option vega
def get_vega(df: pd.DataFrame) -> None:
    df['ImpliedVolatility']: pd.Series = py_vollib_vectorized.vectorized_vega(
        S=df['UnderlyingPrice'], K=df['StrikePrice'], t=df['DTE'] / 365, r=.02,
        flag=df['PutCall'].str[:1], sigma=df['ImpliedVolatility'], q=0, model='black_scholes_merton',
        return_as='numpy'
    )


# ---------------------------------------------------------------------------------------------------- #

# apply each function to each partition
ddf.map_partitions(get_iv)
ddf.map_partitions(get_delta)
ddf.map_partitions(get_vega)

# save each file to specified directory with corresponding ticker name
ddf.to_parquet(OUTPUT_DIR, name_function=(lambda idx: tickers[idx]))
