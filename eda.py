import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


crude_oil = pd.read_csv('data/crude_oil.csv')

date_format = "%Y-%m-%d"
crude_oil['date'] = [datetime.strptime(date, date_format) for date in crude_oil['date']]

# length of dataset
print(len(crude_oil))

print('-----------------------')

# print out descriptive statistics about each column
# for column in crude_oil:
#     print(column)
#     print(crude_oil[column].describe())

# plot the date for each numerical column vs the date for the most recent 3 years
crude_oil_recent = crude_oil[crude_oil['date'] >= '2022-01-01']
columns = crude_oil_recent.columns[1:]
for column in columns:
    # print(f"{column}: {type(crude_oil_recent[column].iloc[0])}")
    # print(crude_oil_recent.dtypes)
    # print(crude_oil_recent.head())
    # break
    plt.plot(crude_oil_recent['date'].values, crude_oil_recent[column].values)
    plt.title(column.upper()+" vs. Date")
    plt.xlabel("Date")
    plt.ylabel(column.upper())
    # plt.savefig('eda_figures/'+'adj_close'+'.png')
    plt.show()