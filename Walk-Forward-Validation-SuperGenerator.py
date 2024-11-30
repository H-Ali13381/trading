"""
Imports and settings
"""
import numpy as np
np.float_ = np.float64

import pandas as pd
from prophet import Prophet
from datetime import timedelta

from sklearn.ensemble import GradientBoostingClassifier

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

"""
Parameters and globals
"""
Results_List = []
Prediction_Dates = []
Predictions = []

country = 'US'
start_date = "1900-01-01"
end_date = None

tickers = ["BTC-USDT"]
is_crypto = True
train_run_sizes = [69]

"""
Import ticker
"""
for ticker in tickers:
    data_original = pd.read_pickle('Data/BTC-USDT-Daily.pkl')
    data_original.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    #data_original['Date'] = data_original['Date'].dt.tz_localize(None)
    
    if is_crypto is False:
        market_holidays_correct = tuple(set(data_original.index))
    
    for train_run_size in train_run_sizes:
        Accuracy_check = []
        for test_run in range(len(data_original)):
            print(f'\tTest run #: {test_run} for {ticker}')
            
            begin = test_run
            end = test_run + train_run_size
            
            data = data_original.copy()[begin:end + 11]
            
            #print(data.shape)
            
            """
            Prophet features
            """
            train_data = data.copy()[['Date','Close']]
            train_data = train_data.rename(columns={'Date':'ds',
                                                    'Close':'y'})
            
            model = Prophet(seasonality_mode = 'multiplicative')
            model.add_country_holidays(country)
            model.fit(train_data)
            
            # Getting last date
            start_date = train_data.iloc[-1,0]
            # Generating future dates
            extra_days_range = 11
            future_days = [start_date + timedelta(days=x) for x in range(1,extra_days_range)]
            # Turning it into a dataframe
            future_days = pd.DataFrame(future_days, columns=['ds'])
            full_timeframe = pd.concat([train_data['ds'], future_days]).reset_index(drop=True)
            
            prophet_output = model.predict(full_timeframe)
            prophet_output = prophet_output[['ds','yhat']]
            prophet_output['yhat derivative'] = np.gradient(prophet_output['yhat'],prophet_output.index)
            prophet_output = prophet_output.rename(columns={"ds":"Date"})
            
            data = pd.merge(data.copy(), prophet_output, on='Date', how='right')
            
            labels = ['Open', 'High', 'Low', 'Volume', 'yhat', 'yhat derivative']
            periods = list(range(1,11))
            
            """
            Shift features
            """
            new_columns = [] #container for shifted columns
            labels = ['Open', 'High', 'Low', 'Volume', 'yhat', 'yhat derivative'] #features to shift
            periods = tuple(range(1,11)) #period of shift
                        
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
            
            # price in 1 days column (forwards)
            label='Close'
            period=1
            new_label = 'Close+1'
            new_columns.append(data[label].shift(-period).rename(new_label))
            
            # concat everything
            data = pd.concat([data] + new_columns, axis=1)

            
            """
            Custom labels for predictions
            """
            new_label_2 = new_label + ' Difference %'
            data[new_label_2] = (data[new_label]-data['Close']) / data['Close'] *100
            
            new_label_3 = new_label + ' Direction'
            data[new_label_3] = np.where(data[new_label_2] > 0, 1, -1)
            
            data.loc[data['Close+1'].isnull(), 'Close+1 Direction'] = None
            data = data.dropna(subset=['Open'])
            
            drop_columns = ['Close+1',
                            'Close+1 Difference %']
            
            data = data.drop(drop_columns, axis=1)
			
            
            """
            Train/Test split
            """
            train_data, test_data = data[:-1], data[-1:]
            train_data = train_data.dropna()
            
            
        
            if train_data.shape[0] != train_run_size:
                break
            
            print(f'train_size: {train_data.shape} \ntest_size: {test_data.shape}\n')
            
            X_train, y_train = train_data.drop(['Date','Close+1 Direction'], axis=1), train_data['Close+1 Direction']
            X_test, y_test = test_data.drop(['Date','Close+1 Direction'], axis=1), test_data['Close+1 Direction']
            
            model = GradientBoostingClassifier(random_state=42)
            model = model.fit(X_train, y_train)
            
            Predictions.extend(model.predict(X_test))
            Prediction_Dates.extend(list(test_data['Date']))
            #Accuracy_check.append(model.score(X_test, y_test))
            
        
            
                
        #Results_List.append (f'Size: {train_run_size} \nTicker: {ticker} \nAccuracy: {sum(Accuracy_check)/len(Accuracy_check) *100} %\n')
        
#print('Score to beat: 55.28%')
#for i in Results_List:
#    print(i)

print('Done---')
    
Final_Predictions_DataFrame = pd.DataFrame({ 'Date': Prediction_Dates,
                                             'Signal': Predictions})
Final_Predictions_DataFrame.to_csv('BTC-USDT-Predictions.csv', index = False)
Final_Predictions_DataFrame.to_pickle('BTC-USDT-Predictions.pkl')