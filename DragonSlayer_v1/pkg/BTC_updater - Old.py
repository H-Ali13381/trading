# -*- coding: utf-8 -*-
"""
BTC-USD info updater

"""

from datetime import datetime, timedelta

import numpy as np
np.float_ = np.float64

import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import GradientBoostingClassifier

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


# Training

def update(country = 'US', ticker = "BTC-USD"):
    
    # Get today's date
    current_dt = datetime.now()
    
    # Calculate the date 180 days ago
    start_dt = str(current_dt - timedelta(days=100)).split()[0]
    current_d = current_dt.strftime('%Y-%m-%d')
    current_dt = current_dt.strptime(current_d,'%Y-%m-%d')
    print(current_dt)

    end_date = None
    train_samples = 69
    data = yf.download(ticker, start=start_dt, end=end_date).reset_index()
    
    # Correcting yfinance Bug ####!!!!!
    data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    data['Date'] = data['Date'].dt.tz_localize(None)

    
    data = data.drop(index=data.index[-1],axis=0) #Exclude current day from train
    data = data.copy()[-train_samples - 11:] #79 rows when including
    
    # Changes
    
    #Train/Test split
    train_data = data.copy()[['Date','Close']]
    train_data = train_data.rename(columns={'Date':'ds',
                                            'Close':'y'})
    
    prophet_model = Prophet(seasonality_mode = 'multiplicative')
    
    prophet_model.add_country_holidays(country)
    prophet_model.fit(train_data)
    
    # Getting last date
    start_date = train_data.iloc[-1,0]
    # Generating future dates
    extra_days_range = 11
    future_days = [start_date + timedelta(days=x) for x in range(1,extra_days_range)]
    
    # Turning prophet results into dataframe
    future_days = pd.DataFrame(future_days, columns=['ds'])
    full_timeframe = pd.concat([train_data['ds'], future_days]).reset_index(drop=True)
    
    prophet_output = prophet_model.predict(full_timeframe)
    prophet_output = prophet_output[['ds','yhat']]
    prophet_output['yhat derivative'] = np.gradient(prophet_output['yhat'],prophet_output.index)
    prophet_output = prophet_output.rename(columns={"ds":"Date"})
    
    # Merging features
    
    data = pd.merge(data.copy(), prophet_output, on='Date', how='right')
    
    # Shifts
    new_columns = [] #container for shifted columns
    labels = ['Open', 'High', 'Low', 'Volume', 'yhat', 'yhat derivative']
    periods = list(range(1,11))  #period of shift
    
    #backwards
    for label in labels:
        for period in periods:
            new_label = label + '-' + str(period)
            new_columns.append(data[label].shift(period).rename(new_label))
    
    #forwards
    labels = ['yhat','yhat derivative']
    for label in labels:
        for period in periods:
            new_label = label + '+' + str(period)
            new_columns.append(data[label].shift(-period).rename(new_label))
    
    # Goal: Predicting price in 1 day
    label='Close'
    period=1
    new_label = 'Close+1'
    new_columns.append(data[label].shift(-period).rename(new_label))
    
    # concat everything
    data = pd.concat([data] + new_columns, axis=1)
    
    # Custom labels for predictions
    
    new_label_2 = new_label + ' Difference %'
    data[new_label_2] = (data[new_label]-data['Close']) / data['Close'] *100
    
    new_label_3 = new_label + ' Direction'
    data[new_label_3] = np.where(data[new_label_2] > 0, 1, -1)
    
    data.loc[data['Close+1'].isnull(), 'Close+1 Direction'] = None
    data = data.dropna(subset=['Open'])
    
    drop_columns = ['Close+1',
                    'Close+1 Difference %']
    
    data = data.drop(drop_columns, axis=1)
    
    train_data = data
    train_data = train_data.dropna()
    X_train, y_train = train_data.drop(['Date','Close+1 Direction'], axis=1), train_data['Close+1 Direction']
    
    train_data = data
    train_data = train_data.dropna()
    X_train, y_train = train_data.drop(['Date','Close+1 Direction'], axis=1), train_data['Close+1 Direction']
    
    model = GradientBoostingClassifier(random_state=42)
    model = model.fit(X_train, y_train)
    
    # Predicting
    
    today = yf.download(ticker, start=start_dt, end=end_date)
    today = today.tail(15)
    today = today.reset_index()
    
    today.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    today['Date'] = today['Date'].dt.tz_localize(None)
    
    today = today.rename(columns={'Date':'ds',
                                  'Close':'y'})
    
    # Getting last date
    start_date = today.iloc[-1,0]
    # Generating future dates
    extra_days_range = 11
    future_days = [start_date + timedelta(days=x) for x in range(1,extra_days_range)]
    
    # Turning it into a dataframe
    future_days = pd.DataFrame(future_days, columns=['ds'])
    full_timeframe = pd.concat([today['ds'], future_days])
    
    train_data = data.copy()[['Date','Close']]
    train_data = train_data.rename(columns={'Date':'ds',
                                            'Close':'y'})
    
    prophet_output = prophet_model.predict(today)
    prophet_output = prophet_model.predict(full_timeframe)
    prophet_output = prophet_output[['ds','yhat']]
    prophet_output['yhat derivative'] = np.gradient(prophet_output['yhat'],prophet_output.index)
    
    today = pd.concat([today, prophet_output], axis=1)
    
    labels = ['Open', 'High', 'Low', 'Volume', 'yhat', 'yhat derivative']
    periods = list(range(1,11))
    
    today = today.rename(columns={'ds':'Date',
                                  'y':'Close'})
    
    new_columns = []
    
    for label in labels:
        for period in periods:
            new_label = label + '-' + str(period)
            new_columns.append(today[label].shift(period).rename(new_label))
    
    labels = ['yhat','yhat derivative']
    for label in labels:
        for period in periods:
            new_label = label + '+' + str(period)
            new_columns.append(today[label].shift(-period).rename(new_label))
    
    today = pd.concat([today] + new_columns, axis=1)
    
    today = today.copy().tail(11)
    today = today.copy().head(1)
    
    drop_columns = ['Date']
    
    today = today.drop(drop_columns, axis=1)
    
    result = model.predict(today)[0]

    # merge
    file_name = './db/' + ticker + '_Predictions.pkl'
    signals_history = pd.read_pickle(file_name)

    df_temp = pd.DataFrame({'Date': current_dt,
               'Signal': result},
                index=[0])
    new_df = pd.concat([signals_history, df_temp], ignore_index=True).drop_duplicates()
    new_df.to_pickle(file_name)
    
    
    file_name = './logs/' + ticker + '_log.txt'
    log_text = 'Update completed on ' + current_d + '\n'
    
    with open(file_name, 'a') as f:
        f.write(log_text)
    print('\nUpdate completed on ' + current_d)