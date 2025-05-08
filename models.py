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

def plot_diff(train_diff_data, title: str):
    """
    Plots and saves the differenced data, the ACF and PACF of the differenced data
    """
    series = train_diff_data['adj_close']
    plt.figure()
    plot_acf(series)
    plt.savefig('model_figures/diff_data_plot/acf/'+title+'_acf.png')
    plt.close()

    plt.figure()
    plot_pacf(series)
    plt.savefig('model_figures/diff_data_plot/pacf/'+title+'_pacf.png')
    plt.close()

    plt.figure()
    series.plot()
    plt.title("ADJ_CLOSE vs. Date: "+title)
    plt.savefig('model_figures/diff_data_plot/raw_diff/'+title+'_raw_diff.png')
    plt.close()

def train_test_model_manual(full_data, train_df, test_df, title: str, p, d, q, folder):
    """
    Fits the arima model to the given subset of training data, saves SARIMAX 
    results to a txt file, saves residual and density of residuals plots, saves
    ACF and PACF plots of the residuals, plots the forecasted results, saves 
    error for test set in text file (mean absolute error, 
    mean absolute percentage errormean squared error)

    p = Autoregressive parameter
    d = differencing parameter (integrated)
    q = moving average parameter
    
    folder = "AR" or "MA"
    """
    # get full series
    series = full_data['adj_close']
    series_train = train_df['adj_close']
    series_test = test_df['adj_close']

    # train model
    model = ARIMA(series_train, order=(p, d, q))
    model_fit = model.fit()

    # save results of model training
    with open(f'results/{title}/{folder}/sarimax.txt','w+') as f:
        f.write(model_fit.summary().as_text())
        f.close()

    # plot residuals and their distribution
    residuals = model_fit.resid[1:]
    plt.figure()
    _, ax = plt.subplots(1,2)
    residuals.plot(title ='Residuals', ax=ax[0])
    residuals.plot(title = 'Density', kind = 'kde', ax=ax[1])
    plt.savefig(f'results/{title}/{folder}/resid_density.png')
    plt.close()

    # forecast and plot
    forecast = model_fit.forecast(len(series_test))
    plot_df = pd.DataFrame({'actual': series})
    plot_df['forecast'] = np.nan
    plot_df.loc[series_test.index, 'forecast'] = forecast.values 
    plt.figure()
    plot_df.plot()
    plt.savefig(f'results/{title}/{folder}/forecast.png')
    plt.close()

    # save error values in txt file
    mae = mean_absolute_error(series_test, forecast)
    mape = mean_absolute_percentage_error(series_test, forecast)
    rmse = np.sqrt(mean_squared_error(series_test, forecast))

    with open(f'results/{title}/{folder}/errors.txt','w+') as f:
        f.write(f'mean absolue error: {mae}\n')
        f.write(f'mean absolute percentage error: {mape}\n')
        f.write(f'residual mean squared error: {rmse}')
        f.close()

    if title == 'full':
        title = '24'

    return int(title), rmse

def train_test_model_auto(full_data, train_df, test_df, title):
    """
    Autofits the arima model and saves the results
    """
    # get full series
    series = full_data['adj_close']
    series_train = train_df['adj_close']
    series_test = test_df['adj_close']

    # train model
    model = pm.auto_arima(series_train, stepwise=False)

    # save results of model training
    with open(f'results/{title}/auto/sarimax.txt','w+') as f:
        f.write(model.summary().as_text())
        f.close()

    # forecast and plot
    forecast = model.predict(n_periods = len(series_test))
    plot_df = pd.DataFrame({'actual': series})
    plot_df['forecast'] = np.nan
    plot_df.loc[series_test.index, 'forecast'] = forecast.values 
    plt.figure()
    plot_df.plot()
    plt.savefig(f'results/{title}/auto/forecast.png')
    plt.close()

    # save error values in txt file
    mae = mean_absolute_error(series_test, forecast)
    mape = mean_absolute_percentage_error(series_test, forecast)
    rmse = np.sqrt(mean_squared_error(series_test, forecast))

    with open(f'results/{title}/auto/errors.txt','w+') as f:
        f.write(f'mean absolue error: {mae}\n')
        f.write(f'mean absolute percentage error: {mape}\n')
        f.write(f'residual mean squared error: {rmse}')
        f.close()

    if title == 'full':
        title = '24'

    return int(title), rmse

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
    plot_raw_data(crude_oil, "full")
    plot_raw_data(crude_oil_3, "3yrs")
    plot_raw_data(crude_oil_5, "5yrs")
    plot_raw_data(crude_oil_10, "10yrs")
    plot_raw_data(crude_oil_20, "20yrs")

    # log transformations
    plot_trans_data(crude_oil, "full")
    plot_trans_data(crude_oil_3, "3yrs")
    plot_trans_data(crude_oil_5, "5yrs")
    plot_trans_data(crude_oil_10, "10yrs")
    plot_trans_data(crude_oil_20, "20yrs")

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
    plot_acf_pacf(train_full, 'full')
    # 20 years
    plot_acf_pacf(train_20, '20')
    # 10 years
    plot_acf_pacf(train_10, '10')
    # 5 years
    plot_acf_pacf(train_5, '5')
    # 3 years
    plot_acf_pacf(train_3, '3')

    # ACF plot shows that the correlation with the lags are high and positive
    # with a slow decline. The PACF plot show a single spike at 1 and then small
    # or no spikes after. These are signs of a simple random walk, a common time
    # series, which is not stationary

    # ADF Test : the null hypothesis is that there is a unit root 
    # (non stationary). This means that if you have a large p value then you 
    # fail to reject the null hypothesis, which suggest that the data is not
    # stationary
    adf_test_full = adfuller(train_full['adj_close'])
    print(f'full p-value: {adf_test_full[1]}') # 0.08 => do not reject null => not stationary

    adf_test_20 = adfuller(train_20['adj_close'])
    print(f'20 years p-value: {adf_test_20[1]}') # 0.03 => can reject null => stationary (no differencing needed)

    adf_test_10 = adfuller(train_10['adj_close'])
    print(f'10 years p-value: {adf_test_10[1]}') # 0.17 => do not reject null => not stationary

    adf_test_5 = adfuller(train_5['adj_close'])
    print(f'full data p-value: {adf_test_5[1]}') # 0.63 => do not reject null => not stationary

    adf_test_3 = adfuller(train_3['adj_close'])
    print(f'full data p-value: {adf_test_3[1]}') # 0.53 => do not reject null => not stationary

    # Step 4: Differencing to make stationary
    train_full_diff = train_full.diff().dropna()
    train_20_diff = train_20.diff().dropna() # differencing not needed but leaving this here in case
    train_10_diff = train_10.diff().dropna()
    train_5_diff = train_5.diff().dropna()
    train_3_diff = train_3.diff().dropna()

    # 3 years
    plot_diff(train_3_diff, "3")
    # 5 years
    plot_diff(train_5_diff, "5")
    # 10 years
    plot_diff(train_10_diff, "10")
    # 20 years
    plot_diff(train_20_diff, "20")
    # all years
    plot_diff(train_full_diff, "full")

    ### ADF TEST FOR DIFFERENCED DATA
    adf_test_full_diff = adfuller(train_full_diff['adj_close'])
    print(f'full p-value: {adf_test_full_diff[1]}') # 6.28205508946529e-23 => reject null, stationary

    adf_test_20_diff = adfuller(train_20_diff['adj_close'])
    print(f'20 years p-value: {adf_test_20_diff[1]}') # 1.266836877616905e-20 => reject null, stationary

    adf_test_10_diff = adfuller(train_10_diff['adj_close'])
    print(f'10 years p-value: {adf_test_10_diff[1]}') # 1.709682877551195e-15 => reject null, stationary

    adf_test_5_diff = adfuller(train_5_diff['adj_close'])
    print(f'full data p-value: {adf_test_5_diff[1]}') # 2.9483532229818093e-27 => reject null, stationary

    adf_test_3_diff = adfuller(train_3_diff['adj_close'])
    print(f'full data p-value: {adf_test_3_diff[1]}') # 1.1984028177152274e-20 => reject null, stationary

    # Step 5: Run models for each subset of data and save the results of the
    # fitted model, residual and density plots, acf and pacf plots, forecasted 
    # data plot, error values (mean absolute error, mean absolute percentage error, 
    # mean squared error)

    # full
    # auto
    auto_full = train_test_model_auto(crude_oil, train_full, test_full, 'full')

    # manual
    ar_full = train_test_model_manual(crude_oil, train_full, test_full, 'full', 4, 1, 0, 'AR')
    ma_full = train_test_model_manual(crude_oil, train_full, test_full, 'full', 0, 1, 4, 'MA')

    # 20 years
    # auto
    auto_20 = train_test_model_auto(crude_oil_20, train_20, test_20, '20')

    # manual
    ar_20 = train_test_model_manual(crude_oil_20, train_20, test_20, '20', 4, 1, 0, 'AR')
    ma_20 = train_test_model_manual(crude_oil_20, train_20, test_20, '20', 0, 1, 4, 'MA')

    # 10 years
    # auto
    auto_10 = train_test_model_auto(crude_oil_10, train_10, test_10, '10')

    # manual
    ar_10 = train_test_model_manual(crude_oil_10, train_10, test_10, '10', 6, 1, 0, 'AR')
    ma_10 = train_test_model_manual(crude_oil_10, train_10, test_10, '10', 0, 1, 6, 'MA')

    # 5 years
    # auto
    auto_5 = train_test_model_auto(crude_oil_5, train_5, test_5, '5')

    # manual
    ar_5 = train_test_model_manual(crude_oil_5, train_5, test_5, '5', 6, 1, 0, 'AR')
    ma_5 = train_test_model_manual(crude_oil_5, train_5, test_5, '5', 0, 1, 6, 'MA')

    # 3 years
    # auto
    auto_3 = train_test_model_auto(crude_oil_3, train_3, test_3, '3')

    # manual
    ar_3 = train_test_model_manual(crude_oil_3, train_3, test_3, '3', 6, 1, 0, 'AR')
    ma_3 = train_test_model_manual(crude_oil_3, train_3, test_3, '3', 0, 1, 6, 'MA')

    # Step 6: Compare the other models to autofitting the models
    auto_errors = [auto_full, auto_20, auto_10, auto_5, auto_3]
    auto_sizes, auto_err = zip(*auto_errors)
    ar_errors = [ar_full, ar_20, ar_10, ar_5, ar_3]
    ar_sizes, ar_err = zip(*ar_errors)
    ma_errors = [ma_full, ma_20, ma_10, ma_5, ma_3]
    ma_sizes, ma_err = zip(*ma_errors)

    plt.figure()
    plt.plot(auto_sizes, auto_err, marker='o', linestyle='-')
    plt.xlabel('Subset Size')
    plt.ylabel('Error')
    plt.title('Error vs Subset Size')
    plt.grid(True)
    plt.savefig('results/errors/autofit.png')
    plt.close()

    plt.figure()
    plt.plot(ar_sizes, ar_err, marker='o', linestyle='-')
    plt.xlabel('Subset Size')
    plt.ylabel('Error')
    plt.title('Error vs Subset Size')
    plt.grid(True)
    plt.savefig('results/errors/ar.png')
    plt.close()

    plt.figure()
    plt.plot(ma_sizes, ma_err, marker='o', linestyle='-')
    plt.xlabel('Subset Size')
    plt.ylabel('Error')
    plt.title('Error vs Subset Size')
    plt.grid(True)
    plt.savefig('results/errors/ma.png')
    plt.close()


if __name__ == "__main__":
    main()