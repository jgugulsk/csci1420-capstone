import numpy as np
import pandas as pd

from datetime import datetime

raw_data = pd.read_csv('data/Crude_Oil_Data.csv')

# lower case all the column names
raw_data.columns = [name.lower() for name in raw_data.columns]
# print(raw_data.columns)

# check the type of all the columns
# for col in raw_data.columns:
#     print(type(raw_data[col][1]))
# all numbers are floats, volume is int, the date is a string, need to convert
# string date into datetime

date_format = "%Y-%m-%d"
raw_data['date'] = [datetime.strptime(date.split()[0], date_format) for date in raw_data['date']]

raw_data.to_csv('data/crude_oil.csv', index=False)
# print(type(raw_data['date'][1]))
# print(raw_data['date'][1])