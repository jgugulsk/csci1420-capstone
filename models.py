import pandas as pd
import numpy as np

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def plot_raw_data(oil_data, title: str):
    """
    Plots the raw data given a subset of time
    """
    adj_close = oil_data.columns[1:][0]
    plt.figure()
    plt.plot(oil_data['date'].values, oil_data[adj_close].values)
    plt.title(adj_close.upper()+" vs. Date")
    plt.xlabel("Date")
    plt.ylabel(adj_close.upper())
    plt.savefig('model_figures/'+title+'_adj_close'+'.png')

def plot_trans_data(oil_data, title: str):
    """
    Plots the log transformation of the data given a subset of time
    """
    adj_close = oil_data.columns[1:][0]
    oil_data = oil_data.copy()
    oil_data[adj_close] = np.log(oil_data[adj_close])
    plt.figure()
    plt.plot(oil_data['date'].values, oil_data[adj_close].values)
    plt.title(adj_close.upper()+" vs. Date")
    plt.xlabel("Date")
    plt.ylabel(adj_close.upper())
    plt.savefig('model_figures/trans_data_plot/'+title+'_adj_close'+'.png') 

def main():
    # get main dataset
    crude_oil = pd.read_csv('data/crude_oil.csv')

    # split into 5 different time periods
    crude_oil_3 = crude_oil[crude_oil['date'] >= '2022-01-01']
    crude_oil_5 = crude_oil[crude_oil['date'] >= '2020-01-01']
    crude_oil_10 = crude_oil[crude_oil['date'] >= '2014-01-01']
    crude_oil_20 = crude_oil[crude_oil['date'] >= '2004-01-01']

    # step 1: plot the raw data and the log transformed data
    #   here I am looking for variance (is it constant throughout time?) and if 
    #   there is a change in variance for the transformed data  
      
    # plot the 5 different subsets
    # raw data
    # plot_raw_data(crude_oil, "full")
    # plot_raw_data(crude_oil_3, "3yrs")
    # plot_raw_data(crude_oil_5, "5yrs")
    # plot_raw_data(crude_oil_10, "10yrs")
    # plot_raw_data(crude_oil_20, "20yrs")

    # log transformations
    # plot_trans_data(crude_oil, "full")
    # plot_trans_data(crude_oil_3, "3yrs")
    # plot_trans_data(crude_oil_5, "5yrs")
    # plot_trans_data(crude_oil_10, "10yrs")
    # plot_trans_data(crude_oil_20, "20yrs")

    # Step 2: make train and test splits of the data

    # full
    train_full = crude_oil[crude_oil['date'] < '2023-01-01']
    test_full = crude_oil[crude_oil['date'] >= '2023-01-01']

    # 20 years
    train_20 = crude_oil_20[crude_oil_20['date'] < '2023-01-01']
    test_20 = crude_oil_20[crude_oil_20['date'] >= '2023-01-01']

    # 10 years
    train_10 = crude_oil_10[crude_oil_10['date'] < '2023-01-01']
    test_10 = crude_oil_10[crude_oil_10['date'] >= '2023-01-01']

    # 5 years
    train_5 = crude_oil_5[crude_oil_5['date'] < '2024-01-01']
    test_5 = crude_oil_5[crude_oil_5['date'] >= '2024-01-01']

    # 3 years
    train_3 = crude_oil_3[crude_oil_3['date'] < '2024-01-01']
    test_3 = crude_oil_3[crude_oil_3['date'] >= '2024-01-01']

    # Step 3: check for stationarity of time series
    #   - plots (observe the ones we have saved above)
    #   - ACF Plot and PACF Plot
    #   - ADF test




if __name__ == "__main__":
    main()