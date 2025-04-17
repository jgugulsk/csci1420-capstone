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
    plt.close()

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
    plt.close()

def plot_acf_pacf(train_data, title: str):
    """
    Plots and saves the acf and pacf figures based on the given training data
    """
    series = train_data['adj_close']
    plt.figure()
    plot_acf(series)
    plt.savefig('model_figures/acf_pacf/'+title+'_acf.png')
    plt.close()

    plt.figure()
    plot_pacf(series)
    plt.savefig('model_figures/acf_pacf/'+title+'_pacf.png')
    plt.close()

def main():
    # get main dataset
    crude_oil = pd.read_csv('data/crude_oil.csv')
    crude_oil['date'] = pd.to_datetime(crude_oil['date'])

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

    # ACF and PACF Plots:

    # full 
    # plot_acf_pacf(train_full, 'full')
    # # 20 years
    # plot_acf_pacf(train_20, '20')
    # # 10 years
    # plot_acf_pacf(train_10, '10')
    # # 5 years
    # plot_acf_pacf(train_5, '5')
    # # 3 years
    # plot_acf_pacf(train_3, '3')

    # ACF plot shows that the correlation with the lags are high and positive
    # with a slow decline. The PACF plot show a single spike at 1 and then small
    # or no spikes after. These are signs of a simple random walk, a common time
    # series, which is not stationary

    # ADF Test : the null hypothesis is that there is a unit root 
    # (non stationary). This means that if you have a large p value then you 
    # fail to reject the null hypothesis, which suggest that the data is not
    # stationary
    # adf_test_full = adfuller(train_full['adj_close'])
    # print(f'full p-value: {adf_test_full[1]}') # 0.08 => do not reject null => not stationary

    # adf_test_20 = adfuller(train_20['adj_close'])
    # print(f'20 years p-value: {adf_test_20[1]}') # 0.03 => can reject null => stationary (no differencing needed)

    # adf_test_10 = adfuller(train_10['adj_close'])
    # print(f'10 years p-value: {adf_test_10[1]}') # 0.17 => do not reject null => not stationary

    # adf_test_5 = adfuller(train_5['adj_close'])
    # print(f'full data p-value: {adf_test_5[1]}') # 0.63 => do not reject null => not stationary

    # adf_test_3 = adfuller(train_3['adj_close'])
    # print(f'full data p-value: {adf_test_3[1]}') # 0.53 => do not reject null => not stationary

    






if __name__ == "__main__":
    main()