# Data Collection and Updating CSV files
import ssl
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
import requests
import json
import csv
from bs4 import BeautifulSoup

file = 'ITK Cases.csv'
# ignore ssl errors when retrieving HTML or JSON files
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# create list of all dates up that day
current = datetime.now()
df = pd.read_csv(file)
date, month, year = map(int, df.iloc[-1]['Date'].split('/'))
last_updated = datetime(year, month, date)
dates_for_update = pd.date_range(last_updated+timedelta(days=1), current, freq='d')
dates_for_update = list(dates_for_update.to_pydatetime())
date_list = []
for date in dates_for_update:
    new_date, new_month, new_year = str(int(date.strftime('%d'))), str(int(date.strftime('%m'))), date.strftime('%Y')
    date_list.append(f'{new_date}/{new_month}/{new_year}')


## data retrieval/collection process

# get all the STI and SPY closing prices
stocks_data = yf.download(['^STI', 'SPY'], start=dates_for_update[0], end=dates_for_update[-1])
stock_closing_price = stocks_data['Close']
stock_closing_price.columns = ['STI', 'SPY']
stock_closing_price.reset_index(inplace=True)
print(stock_closing_price, type(stock_closing_price))
print(type(stock_closing_price.iloc[0]['Date']))
#print(stock_closing_price)


# Google Trends Info Retrieval
pytrend = TrendReq(hl='en-GB', tz=360)
keywords = ['child abuse']
monthly = []

# seperate data monthly here to ensure that there will always be data to retrieve, some instances of updating had too
# little days and therefore they cannot get the relative data
def seperate_monthly(lst):
    if len(lst) > 30:
        s_date, s_month, s_year = str(int(lst[0].strftime('%d'))),\
                                  str(int(lst[0].strftime('%m'))), lst[0].strftime('%Y')
        e_date, e_month, e_year = str(int(lst[30].strftime('%d'))),\
                                  str(int(lst[30].strftime('%m'))), lst[30].strftime('%Y')
        time_range = f'{s_year}-{s_month}-{s_date} {e_year}-{e_month}-{e_date}'
        monthly.append((time_range, 30))
        return seperate_monthly(lst[30:])
    else:
        s_date, s_month, s_year = str(int(lst[0].strftime('%d'))), \
                                  str(int(lst[0].strftime('%m'))), lst[0].strftime('%Y')
        fake_start = lst[-1] - timedelta(days=30)
        fake_date, fake_month, fake_year = str(int(fake_start.strftime('%d'))), \
                                  str(int(fake_start.strftime('%m'))), fake_start.strftime('%Y')
        e_date, e_month, e_year = str(int(lst[-1].strftime('%d'))), \
                                  str(int(lst[-1].strftime('%m'))), lst[-1].strftime('%Y')
        time_range = f'{s_year}-{s_month}-{s_date} {e_year}-{e_month}-{e_date}'
        fake_range = f'{fake_year}-{fake_month}-{fake_date} {e_year}-{e_month}-{e_date}'
        monthly.append((fake_range, time_range))
seperate_monthly(dates_for_update)

# reversing the monthly effect if duration to update is < 30 days
GTI_df = None
for month in monthly:
    if type(month) is tuple:
        pytrend.build_payload(kw_list=keywords, timeframe=month[0], geo='SG')
        monthly_data = pytrend.interest_over_time()
        range_val = month[1].split()
        actual_data = monthly_data[range_val[0]:range_val[1]]
    else:
        pytrend.build_payload(kw_list=keywords, timeframe=month, geo='SG')
        actual_data = pytrend.interest_over_time()

    if GTI_df:
        GTI_df = pd.concat([GTI_df, actual_data])
    else:
        GTI_df = actual_data

GTI_df.reset_index(inplace=True)
GTI_df = GTI_df.drop(labels=['isPartial'], axis=1)
print(GTI_df, type(GTI_df))
print(type(GTI_df.iloc[0]['date']))

# Accessing data.gov.sg API and retrieving package info
all_years_involved = list(range(dates_for_update[0].year, dates_for_update[-1].year + 1))
ph_package_url = "https://data.gov.sg/api/3/action/package_show?id=singapore-public-holidays"
ph_data = requests.post(ph_package_url).json()
all_ph_datasets = ph_data['result']['resources']
csv_url = [] # contains all CSV files with PH data of years involved in updating
for dataset in all_ph_datasets:
    for year in all_years_involved:
        if str(year) in dataset['name']:
            csv_url.append(dataset['url'])
            break
    if len(csv_url) == len(all_years_involved):
        break

# getting list of PH from yearly CSV files
ph_list = []
for file in csv_url:
    response = requests.get(file).content.decode('UTF-8')
    ph_data = csv.reader(response.splitlines(), delimiter=',')
    for row in list(ph_data)[1:]:
        ph_list.append(row[0])
print(ph_list, type(ph_list))

# Unemployment scraping from MOM website (for simplicity, will just use the latest value upon updating, consider pushing
# date specific retrieval in future versions/ if API is available)

# Getting all tags of .aspx extension
req_obj = requests.Session()
mom_website = 'https://stats.mom.gov.sg/Pages/Unemployment-Summary-Table.aspx'
req_soup = requests.get(mom_website)
soup = BeautifulSoup(req_soup.content , "lxml")
all_tags = soup.find_all('tr')

# parsing .aspx data and filtering out noise info
all_values = map(lambda row: row.split('\n'), [tag.text for tag in all_tags])
final_val = []
for lst in all_values:
    for value in lst:
        if value == 'n.a.':
            final_val.append(value)
        try:
            flt = float(value)
            final_val.append(flt)
        except ValueError:
            continue
index = final_val.index(next(x for x in final_val if x < 2000))
final_val = final_val[::-1]
final_val = list(filter(lambda x: type(x) == str or x < 100, final_val))

# find index of latest data and retrieve corresponding data
def get_latest():
    quarter_count = 0
    current_pos = 0
    while True:
        if final_val[current_pos] == 'n.a.':
            current_pos += index
            quarter_count += 1
            if quarter_count == 4:
                return final_val[1]
        else:
            return final_val[current_pos]
unemployment = get_latest()


# get temperature and humidity info from API
date_feeder = []
daily_avg_temp = []
daily_rainfall = []
for date in dates_for_update:
    dd, mm, yyyy = date.strftime('%d'), date.strftime('%m'), date.strftime('%Y')
    str_date = f'{yyyy}-{mm}-{dd}'
    temperature_url = 'https://api.data.gov.sg/v1/environment/air-temperature?date=' + str_date
    temp_json = requests.get(temperature_url).json()
    rainfall_url = 'https://api.data.gov.sg/v1/environment/rainfall?date=' + str_date
    rainfall_json = requests.get(temperature_url).json()
    temp_readings_in_date = []
    rainfall_readings_in_date = []
    for timestamp in temp_json['items']:
        all_station_readings = list(map(lambda station: station['value'], timestamp['readings']))
        timestamp_avg = sum(all_station_readings)/len(all_station_readings)
        temp_readings_in_date.append(timestamp_avg)
    last_timestamp = rainfall_json['items'][-1]
    day_rainfall = last_timestamp['readings'][0]['value']
    day_avg_temp = round(sum(temp_readings_in_date)/len(temp_readings_in_date), 1)
    daily_avg_temp.append(day_avg_temp)
    daily_rainfall.append(day_rainfall)
print(daily_avg_temp, type(daily_avg_temp))
print(daily_rainfall, type(daily_rainfall))


# get covid data
covid_url = " http://corona-api.com/countries/SG"
covid_data = requests.get(covid_url).json()
d = json.dumps(covid_data, indent=4)
date_feeder = []
cases = []

# getting date, new cases pairings and sorting them in chronological order
dates_for_update = pd.date_range(datetime(2021, 6, 20), datetime(2021, 7, 25), freq='d')
for date in dates_for_update:
    dd, mm, yyyy = date.strftime('%d'), date.strftime('%m'), date.strftime('%Y')
    date_feeder.append(f'{yyyy}-{mm}-{dd}')
for day in covid_data['data']['timeline']:
    if day['date'] in date_feeder:
        cases.append((day['date'], day['new_confirmed']))
cases.sort(key=lambda datapoint: datapoint[0])
print(cases, type(cases))


## compiling all data into 1 dataframe

# def and implement function to use last data point if NaN (copy and paste below later when collating data)
replica = stock_closing_price.copy()
def retrieve(date, column):
    if date in replica.Date:
        return replica.loc[replica['Date'] == date][column]
    else:
        if date == dates_for_update[0]:
            column = column.lower()
            return df.iloc[-1][column]
        else:
            return retrieve(date-timedelta(days=1), column)

#creating new datafrane to write into csv
new_rows = pd.DataFrame()
new_rows['Date'] = dates_for_update
new_rows['ITK'] = 0
new_rows['sti'] = new_rows.apply(lambda row: retrieve(row['Date'], 'STI'), axis=1) # bug: recursion not done properly
print(new_rows['sti'])

