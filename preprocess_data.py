import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
def filter_marketcap(data,last_day,top_n):
    df = pd.read_csv(data,index_col = 0,parse_dates = [0])
    df = df[df.index <= last_day]
    df_last = df.iloc[-1,:]
    df_sorted = df_last.nlargest(top_n)
    return df_sorted.index
    

def calculateEma(series, period, keep_length= True):
    ema = []
    num_ticker = series.shape[1]
    empty = [0 for _ in range(num_ticker)]
    if keep_length:
        ema = [empty for _ in range(period - 1)]
    # print(ema)

    #get n sma first and calculate the next n period ema
    sma = sum(series[:period]) / period
    multiplier = 2 / float(1 + period)
    ema.append(sma)
    j = len(ema)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(((series[period] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in series[period+1:]:
        tmp = ((i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)
    return np.asarray(ema, dtype= np.float32)

def prepair_data(path, marketcap, last_date, n,window_x,window_y, close=['null','interpolate','ffill'], vol=['null','interpolate','fill0']):
    df = pd.read_csv(path)
    df['date'] = df.date.apply(pd.Timestamp)
    df['dow'] = df.date.apply(lambda x: x.dayofweek)
    ## just select working days
    df = df[(df.dow<=4)&(df.dow>=0)]
    df = df.drop(['dow'],axis=1)
    df = df.pivot_table(index='date', columns='ticker')
    ## select tickers not nan in final day
#     columns = df.close.columns[~df.close.iloc[-1].isna()]
# select tickers in top_n marketcap
    columns = filter_marketcap(marketcap,last_date,n)
    df = df.iloc[:, df.columns.get_level_values(1).isin(columns)]
    #fill missing data
    for m in close:
        if (m =='interpolate'):
            df.close = df.close.interpolate(method='linear',limit_area='inside',limit_direction='both', axis=0)
        elif (m == 'ffill'):
            df.close = df.close.ffill()
        elif (m == 'null'):
            continue
        else:
          print('Error: Please enter the correct method of fill missing data')
    
    for m in vol:
        if (m =='interpolate'):
            df.volume = df.volume.interpolate(method='linear',limit_area='inside',limit_direction='both', axis=0)
        elif (m == 'fill0'):
            df.volume = df.volume.fillna(0)
        elif (m == 'null'):
            continue
        else:
          print('Error: Please enter the correct method of fill missing data')
    
    close = df.close
    daily_return = ((close.shift(-1) - close)/close).shift(1)
    tickers = df.close.columns
    X = df.values.reshape(df.shape[0],2,-1)
    y = daily_return.values
    ## fill X
    ##fill nan by 0.0
    X[np.isnan(X)] = 0.0
    ## fill y
    y[np.isnan(y)] = -1e2
    # y[np.isnan(y)] = 0
    # X1 = rolling_array(X[window_x:],stepsize=1,window=window_y)
    X = rolling_array(X[:-window_y],stepsize=1,window=window_x)
    y = rolling_array(y[window_x:],stepsize=1,window=window_y)
    X = np.moveaxis(X,-1,1)
    # X1 = np.moveaxis(X1,-1,1)
    y = np.swapaxes(y,1,2)
    return X,y,tickers


# def prepair_data(path, marketcap, last_date, n,window_x,window_y,periods, close= ['interpolate','ffill'], vol= ['interpolate','fill0'], training= True):
#     df = pd.read_csv(path)   
#     df['date'] = df.date.apply(pd.Timestamp)
#     df['dow'] = df.date.apply(lambda x: x.dayofweek)
#     ## just select working days
#     df = df[(df.dow<=4)&(df.dow>=0)]
#     df = df.drop(['dow'],axis=1)
#     df = df.pivot_table(index='date', columns='ticker')

#     ## select tickers not nan in final day
# #     columns = df.close.columns[~df.close.iloc[-1].isna()]
# # select tickers in top_n marketcap
#     columns = filter_marketcap(marketcap,last_date,n)
#     df = df.iloc[:, df.columns.get_level_values(1).isin(columns)]
#     #fill missing data
#     for m in close:
#         if (m =='interpolate'):
#             df.close = df.close.interpolate(method='linear',limit_area='inside',limit_direction='both', axis=0)
#         elif (m == 'ffill'):
#             df.close = df.close.ffill()
#         else:
#           print('Error: Please enter the correct method of fill missing data')
    
#     for m in vol:
#         if (m =='interpolate'):
#             df.volume = df.volume.interpolate(method='linear',limit_area='inside',limit_direction='both', axis=0)
#         elif (m == 'fill0'):
#             df.volume = df.volume.fillna(0)
#         else:
#           print('Error: Please enter the correct method of fill missing data')

#     close = df.close
#     daily_return = ((close.shift(-1) - close)/close).shift(1)
#     daily_return = daily_return.fillna(0)
# #     daily_return = (close.apply(lambda x: np.log(x) - np.log(x.shift(1)))).iloc[1:]
# #     for m in dailyreturn:
# #         if (m =='interpolate'):
# #             daily_return = daily_return.interpolate(method='linear',limit_area="inside",limit_direction='both', axis=0)
# #         elif (m == 'fill0'):
# #             daily_return = daily_return.fillna(0)
# #         else:
# #             print('Error: Please enter the correct method of fill missing data')

#     # daily_return = daily_return.fillna(daily_return.min(axis=0),axis=0)
#     # daily_return = daily_return.fillna(daily_return.min(axis=0),inplace=True)
#     # daily_return.fillna(daily_return.min(axis=0), inplace=True)

#     tickers = df.close.columns

#     X = df.values.reshape(df.shape[0],2,-1)
#     y = daily_return.values

#     ## fill X
#     ##fill nan by 0.0
#     X[np.isnan(X)] = 0.0
#     ## fill y
#     y[np.isnan(y)] = -1e2
#     # y[np.isnan(y)] = 0
    
#     if isinstance(periods, int):
#         periods = [periods]
#         max_period = max(periods)
#     if max_period != 0:
#         for period in periods:
#             ema  = calculateEma(close, period)
#             X = np.concatenate((X, ema[:, np.newaxis, :]), axis=1)
    
#     X = X[max_period:]
#     y = y[max_period:]
#     if training:
#         X = rolling_array(X[:-window_y],stepsize=1,window=window_x)
#         y = rolling_array(y[window_x:],stepsize=1,window=window_y)
#         y = np.swapaxes(y,1,2)
#     # X1 = rolling_array(X[window_x:],stepsize=1,window=window_y)

#     X = rolling_array(X[:-window_y],stepsize=1,window=window_x)
#     y = rolling_array(y[window_x:],stepsize=1,window=window_y)
#     X = np.moveaxis(X,-1,1)
#     # X1 = np.moveaxis(X1,-1,1)
#     y = np.swapaxes(y,1,2)

#     return X,y,tickers

def rolling_array(a, stepsize=1, window=60):
    n = a.shape[0]
    return np.stack((a[i:i + window:stepsize] for i in range(0,n - window + 1)),axis=0)
