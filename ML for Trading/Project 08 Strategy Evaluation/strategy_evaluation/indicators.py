import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import matplotlib.pyplot as plt



def author():
    return 'Shrikanth Mahale'

def get_indicators(indicator, look_back_window , prices, Symbol):
    df_prices = prices.copy()
    if indicator == 'SMA':
        normed_prices = prices / prices.iloc[0]
        sma = prices.rolling(window=look_back_window, min_periods=1).mean()
        psma = prices / sma
        df_prices['SMA'] = sma
        return sma,psma
    elif indicator == 'Bollinger':
        sma = prices.rolling(window=look_back_window,center=False).mean()
        rolling_std = prices.rolling(window=look_back_window, min_periods=look_back_window).std()
        top_bolinger_band = sma + (2 * rolling_std)
        bottom_bolinger_band = sma - (2 * rolling_std)
        bbp = (prices - bottom_bolinger_band) / (top_bolinger_band - bottom_bolinger_band)
        return bbp
    elif indicator == 'Momentum':
        Momentum=(prices/prices.shift(look_back_window))-1
        df_prices['Momentum'] = Momentum
        return Momentum
    elif indicator == 'Volatility':
        DailyReturns = (prices/prices.shift(1)) - 1
        Volatility = DailyReturns.rolling(window = look_back_window, min_periods = look_back_window).std()
        df_prices['Volatility'] = Volatility
        df_prices.rename(columns={Symbol[0]:"Prices"}, inplace = True)
        return Volatility
    elif indicator == 'CCI':
        rolling_mean = prices.rolling(window=look_back_window,center=False).mean()
        cci = (prices-rolling_mean)/(2.5 * prices.std())
        df_prices['CCI'] = cci
        return cci
    elif indicator == 'EMA':
        ema = prices.ewm(min_periods = look_back_window, com = look_back_window).mean()
        df_prices['EMA'] = ema
        return ema
    
    

def execute_indicators():
    syms = ['JPM']
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    dates = pd.date_range(start_date, end_date)
    prices = util.get_data(syms, dates)
    prices.fillna(method="ffill", inplace=True)  #Forward fill of null values
    prices.fillna(method="bfill", inplace=True)  #Backward fill of null values
    prices = prices[syms]
    normed_prices = prices / prices.iloc[0]
    
    SMA,PSMA = get_indicators('SMA',14,prices,syms)
    Bollinger = get_indicators('Bollinger',14,prices,syms)
    Momentum = get_indicators('Momentum',14,prices,syms)
    Volatility = get_indicators('Volatility',14,prices,syms)
    CCI = get_indicators('CCI',14,prices,syms)
    EMA = get_indicators('EMA',14,prices,syms)


if __name__ == "__main__":
    execute_indicators()


    