import pandas as pd
import numpy as np
import datetime as dt
import util
import matplotlib.pyplot as plt



def author():
    return 'Shrikanth Mahale'

def get_indicators(indicator, look_back_window , prices, Symbol):
    df_prices = prices.copy()
    if indicator == 'SMA':
        sma = prices.rolling(window = look_back_window, min_periods = look_back_window).mean()
        psma = sma / prices
        df_prices['SMA'] = sma
        df_prices['SMA/PRICE'] = psma
        df_prices.rename(columns={Symbol[0]:"Prices"}, inplace = True)
        Figure = df_prices.plot(figsize=(10, 7), title = 'SMA').get_figure()
        Figure.savefig('Price - SMA.png')
        return df_prices
    elif indicator == 'Bollinger':
        sma = prices.rolling(window = look_back_window, center=False).mean()
        rolling_std = prices.rolling(window = look_back_window, min_periods = look_back_window).std()
        top_bolinger_band = sma + (2 * rolling_std)
        bottom_bolinger_band = sma - (2 * rolling_std)
        BBValue = (df_prices[Symbol] - sma)/(2 * rolling_std)
        df_prices['BBValue'] = BBValue
        df_prices['SMA'] = sma
        df_prices['Top_Bollinger_Band'] = top_bolinger_band
        df_prices['Bottom_Bollinger_Band'] = bottom_bolinger_band
        df_prices.rename(columns={Symbol[0]:"Prices"}, inplace = True)
        Figure = df_prices.plot(figsize=(10, 7), title = 'Bollinger Bands').get_figure()
        Figure.savefig('Bollinger Bands.png')
        return df_prices
    elif indicator == 'Momentum':
        Momentum=(prices/prices.shift(look_back_window))-1
        df_prices['Momentum'] = Momentum
        df_prices.rename(columns={Symbol[0]:"Prices"}, inplace = True)
        Figure = df_prices.plot(figsize=(10, 7), title = 'Momentum').get_figure()
        Figure.savefig('Momentum.png')
        return df_prices
    elif indicator == 'Volatility':
        DailyReturns = (prices/prices.shift(1)) - 1
        Volatility = DailyReturns.rolling(window = look_back_window, min_periods = look_back_window).std()
        df_prices['Volatility'] = Volatility
        df_prices.rename(columns={Symbol[0]:"Prices"}, inplace = True)
        Figure = df_prices.plot(figsize=(10, 7), title = 'Volatility').get_figure()
        Figure.savefig('Volatility.png')
        return df_prices
    elif indicator == 'CCI':
        rolling_mean = prices.rolling(window=look_back_window,center=False).mean()
        cci = (prices-rolling_mean)/(2.5 * prices.std())
        df_prices['CCI'] = cci
        df_prices.rename(columns={Symbol[0]:"Prices"}, inplace = True)
        Figure = df_prices.plot(figsize=(10, 7), title = 'CCI').get_figure()
        Figure.savefig('CCI.png')
        return df_prices
    elif indicator == 'EMA':
        ema = prices.ewm(min_periods = look_back_window, com = look_back_window).mean()
        df_prices['EMA'] = ema
        df_prices.rename(columns={Symbol[0]:"Prices"}, inplace = True)
        Figure = df_prices.plot(figsize=(10, 7), title = 'EMA').get_figure()
        Figure.savefig('EMA.png')
        return df_prices
    

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

    
    SMA = get_indicators('SMA',21,normed_prices,syms)
    Bollinger = get_indicators('Bollinger',21,normed_prices,syms)
    Momentum = get_indicators('Momentum',21,normed_prices,syms)
    Volatility = get_indicators('Volatility',21,normed_prices,syms)
    CCI = get_indicators('CCI',21,normed_prices,syms)
    EMA = get_indicators('EMA',21,normed_prices,syms)


if __name__ == "__main__":
    execute_indicators()


    