# ML Model and Prediction Maker

import numpy as np
import pandas as pd
import math
import datetime
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import matplotlib


eval_selector = input('Intake(ITK) or Intervention(ITV)? Please key in 3 letter code: ').upper()
file = f"{eval_selector} Cases.csv"
df = pd.read_csv(file)

# obtaining primary statistics (not for use in this model, just for understanding purposes)
def primary_stats(dataframe):
    stats = {'Mean' : round(dataframe[eval_selector].mean(), 2),
             'Median' : round(dataframe[eval_selector].median(), 2),
             'Min' : round(dataframe[eval_selector].min(), 2),
             'Max' : round(dataframe[eval_selector].max(), 2)}
    for key, value in stats.items():
        print(key, str(value))
primary_stats(df)

# converting excel date data into datetime object
def convert_to_date(date_string):
    date, month, year = list(map(int, date_string.split('/')))
    return datetime.datetime(year, month, date)
df['Date'] = df.apply(lambda row: convert_to_date(row['Date']), axis=1)

# splitting train and test cases
split_date = '31-Dec-2020'
itk_train = df.loc[df['Date'] <= split_date].copy()
itk_test = df.loc[df['Date'] > split_date].copy()

# defining all feature creation functions
df_copy = df.copy()
def day_mean(index, days, label=None):
    if (index >= (days + 1)):
        new_data = df_copy.iloc[index-days+1:index+1]
    else:
        new_data = df_copy.iloc[:index+1]
    return new_data[label].mean()

def day_max(index, days, label=None):
    if (index >= (days + 1)):
        new_data = df_copy.iloc[index-days+1:index+1]
    else:
        new_data = df_copy.iloc[:index+1]
    return new_data[label].max()

def day_min(index, days, label=None):
    if (index >= (days + 1)):
        new_data = df_copy.iloc[index-days+1:index+1]
    else:
        new_data = df_copy.iloc[:index+1]
    return new_data[label].min()

def day_med(index, days, label=None):
    if (index >= (days + 1)):
        new_data = df_copy.iloc[index-days+1:index+1]
    else:
        new_data = df_copy.iloc[:index+1]
    return new_data[label].median()

def day_std(index, days, label=None):
    if (index >= (days + 1)):
        new_data = df_copy.iloc[index-days+1:index+1]
    else:
        new_data = df_copy.iloc[:index+1]
    return (0 if math.isnan(new_data[label].std()) else new_data[label].std())


# time series feature creation and selection for test and train datasets
selected_features_rfe = None
def create_features(df):
    # creates time series features
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear
    df[f'{eval_selector} 7 day avg'] = df.apply(lambda row: day_mean(row.name, 7, eval_selector), axis=1)
    df['spy 7 day max'] = df.apply(lambda row: day_max(int(row.name), 7, 'spy'), axis=1)
    df['GTI 35 day mean'] = df.apply(lambda row: day_mean(int(row.name), 35, 'Google Trends Index'), axis=1)
    df['spy 30 day median'] = df.apply(lambda row: day_med(int(row.name), 30, 'spy'), axis=1)
    df['spy 30 day std'] = df.apply(lambda row: day_std(int(row.name), 30, 'spy'), axis=1)
    df['temperature 35 day std'] = df.apply(lambda row: day_std(int(row.name), 35, 'Temperature'), axis=1)
    df['temperature 30 day median']= df.apply(lambda row: day_med(int(row.name), 30, 'Temperature'), axis=1)
    df['spy 7 day mean'] = df.apply(lambda row: day_mean(int(row.name), 7, 'spy'), axis=1)
    df['GTI 30 day max'] = df.apply(lambda row: day_max(int(row.name), 30, 'Google Trends Index'), axis=1)
    df['rainfall 14 day max'] = df.apply(lambda row: day_max(int(row.name), 14, 'Rainfall'), axis=1)
    df['temperature 30 day min'] = df.apply(lambda row: day_min(int(row.name), 30, 'Temperature'), axis=1)
    df['sti 30 day std'] = df.apply(lambda row: day_std(int(row.name), 30, 'sti'), axis=1)

    # check feature importance (this portion is buggy, may want to readjust things below)
    test_df = df.copy()
    ftest_y = test_df['ITK']
    test_df = test_df.drop([eval_selector, 'Date'], axis=1)
    test_df.drop(test_df.columns[test_df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    nof_list = np.arange(1, len(test_df.columns))
    nof_score = []
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(test_df)
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(test_df, ftest_y, test_size=0.3
                                                            , random_state=0,
                                                            shuffle=False)
        ftest_model = LinearRegression()
        rfe = RFE(ftest_model, nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
        ftest_model.fit(X_train_rfe, y_train)
        score = ftest_model.score(X_test_rfe, y_test)
        nof_score.append(score)
    scores = list(enumerate(nof_score))
    scores.sort(key=lambda x: x[1], reverse=True)
    highest_raw_score = scores[0][1]
    scores = list(filter(lambda x: highest_raw_score - x[1] <= 0.02, scores))
    scores.sort(key=lambda x: x[0], reverse=True)
    optimum_features, highest_adj_score = scores[0]
    optimum_features += 1
    print("Optimum number of features: %d" % optimum_features)
    print("Score with %d features: %f" % (optimum_features, highest_adj_score))

    # Dynamically extract the most optimum features
    cols = list(test_df.columns)
    ftest_model = LinearRegression()
    rfe = RFE(ftest_model, optimum_features)
    X_rfe = rfe.fit_transform(test_df, ftest_y)
    ftest_model.fit(X_rfe, ftest_y)
    temp = pd.Series(rfe.support_, index=cols)
    global selected_features_rfe
    selected_features_rfe = temp[temp == True].index
    selected_features_rfe = selected_features_rfe.tolist()
    print(selected_features_rfe)
    all_features = [eval_selector, 'Date']
    all_features.extend(selected_features_rfe)

    X = df[all_features]
    X.set_index(['Date'], inplace=True)
    return X

X_train = create_features(itk_train)
X_test = create_features(itk_test)

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #print(X_train)
    #print(y_train)

# adfuller test for stationarity (required because Time Series Model assumes that all features are stationary)
def adf_test(ds):
    dftest = adfuller(ds, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags','# Observations'])

    for key, value in dftest[4].items():
       adf['Critical Value (%s)'%key] = value
    print (adf)

    p = adf['p-value']
    if p <= 0.05:
        print("\nSeries is Stationary")
    else:
        print("\nSeries is Non-Stationary")

for i in X_train.columns:
    print("Column: ",i)
    print('--------------------------------------')
    adf_test(X_train[i])
    print('\n')

X_differenced = X_train.diff().dropna()

for i in X_differenced.columns:
    print("Differenced Column: ",i)
    print('--------------------------------------')
    adf_test(X_differenced[i])
    print('\n')

# Fitting into Time Series Model
model = VAR(X_differenced)
results = model.fit()
results.summary()

# Forecasting for n steps ahead
lag_order = results.k_ar
predicted = results.forecast(X_differenced.values[-lag_order:], len(X_test))
print(type(predicted))
print(predicted)
forecast = pd.DataFrame(predicted, index = X_test.index, columns = X_test.columns)

# to undo the differencing done earlier to achieve stationarity
def invert_transformation(ds, df_forecast, second_diff=False):
    for col in ds.columns:
        # Undo the 2nd Differencing
        if second_diff:
            df_forecast[str(col)] = (ds[col].iloc[-1] - ds[col].iloc[-2]) + df_forecast[str(col)].cumsum()

        # Undo the 1st Differencing
        df_forecast[str(col)] = ds[col].iloc[-1] + df_forecast[str(col)].cumsum()

    return df_forecast

forecast_values = invert_transformation(X_train, forecast)
print(forecast_values)

# MAE for Time Series
for column in X_test.columns:
    ts_mae = mean_absolute_error(X_test[column], forecast_values[column])
    print(f"Time Series Mean Absolute Error for {column}: ", ts_mae)

## Gradient Boosting using XGBoost
# creating new dataset
new_df = X_test[selected_features_rfe]

new_df['Predicted cases'] = forecast_values[eval_selector]
new_df['Date'] = new_df.index
new_df['length'] = list(range(len(new_df)))
new_df.set_index(['length'], inplace=True)
print(new_df)

# split train/test cases
X_test['Date'] = X_test.index
split_date = datetime.datetime(2021, 5, 31)
xgb_train = new_df.loc[new_df['Date'] <= split_date].copy()
xgb_test = new_df.loc[new_df['Date'] > split_date].copy()
xgb_train.set_index(['Date'], inplace=True)
xgb_test.set_index(['Date'], inplace=True)

# calculating error in prediction

def difference(index, forecasted):
    actual = X_test.loc[index][eval_selector]
    return actual-forecasted

xgb_y_train = xgb_train.apply(lambda row: difference(row.name, row['Predicted cases']), axis=1)
xgb_y_test = xgb_test.apply(lambda row: difference(row.name, row['Predicted cases']), axis=1)
actual_copy = X_test.loc[X_test['Date'] > split_date].copy()
actual_cases = actual_copy[eval_selector]

# Error Reduction Model
reg = XGBRegressor()
new = reg.fit(xgb_train, xgb_y_train)
pred = new.predict(xgb_test)
xgb_test['Error Correction'] = pred
xgb_test['Adjusted Predictions'] = xgb_test.apply(lambda row: int(row['Predicted cases'] + row['Error Correction']),
                                                  axis=1)
print(xgb_test['Adjusted Predictions'][:-11], actual_cases)

# MAE test
xgb_mae = mean_absolute_error(xgb_test['Adjusted Predictions'], actual_cases)
xgb_mape = mean_absolute_percentage_error(xgb_test['Adjusted Predictions'], actual_cases)
xgb_rsme = mean_squared_error(xgb_test['Adjusted Predictions'], actual_cases, squared=False)

print("After XGB Mean Absolute Error: ", xgb_mae)
print("After XGB Mean Absolute Percentage Error: ", xgb_mape)
print("After XGB R-Squared Mean Error: ", xgb_rsme)

# can include additional code to export results to csv
