import pandas as pd
import numpy as np
import datetime as dt
import util
import matplotlib.pyplot as plt


def get_indicators(indicator, look_back_window , prices, Symbol):
    df_prices = prices.copy()
    if indicator == 'SMA':
        sma = prices.rolling(window = look_back_window, center=False).mean()
        psma = sma / prices
        df_prices['SMA'] = sma
        df_prices['PSMA'] = psma
        df_prices.plot(figsize=(10, 7))
        return df_prices
    elif indicator == 'Bollinger':
        sma = prices.rolling(window = look_back_window, center=False).mean()
        rolling_std = prices.rolling(window = look_back_window, min_periods = look_back_window).std()
        top_bb = sma + (2 * rolling_std)
        bottom_bb = sma - (2 * rolling_std)
        Bollinger = (prices - bottom_bb) / (top_bb - bottom_bb)
        df_prices['Bollinger_Band'] = Bollinger
        df_prices['Top_Bollinger_Band'] = top_bb
        df_prices['Bottom_Bollinger_Band'] = bottom_bb
        df_prices.plot(figsize=(10, 7))
        return df_prices
    elif indicator == 'Momentum':
        Momentum=(prices/prices.shift(look_back_window))-1
        df_prices['Momentum'] = Momentum
        df_prices.plot(figsize=(10, 7))
        return df_prices
    elif indicator == 'Volatility':
        DailyReturns = (prices/prices.shift(1)) - 1
        Volatility = DailyReturns.rolling(window = look_back_window, min_periods = look_back_window).std()
        df_prices['Volatility'] = Volatility
    elif indicator == 'CCI':
        rm = prices.rolling(window=look_back_window,center=False).mean()
        cci = (prices-rm)/(2.5 * prices.std())
        df_prices['CCI'] = cci
        df_prices.plot(figsize=(10, 7))
        return df_prices

def test_code():
    syms = ['JPM']
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    dates = pd.date_range(start_date, end_date)
    prices = util.get_data(syms, dates)
    prices.fillna(method="ffill", inplace=True)  #Forward fill of null values
    prices.fillna(method="bfill", inplace=True)  #Backward fill of null values
    prices = prices[syms]
    normed_prices = prices / prices.iloc[0]
    
    SMA = get_indicators('SMA',14,normed_prices,syms)
    Bollinger = get_indicators('Bollinger',14,normed_prices,syms)
    Momentum = get_indicators('Momentum',14,normed_prices,syms)
    Volatility = get_indicators('Volatility',14,normed_prices,syms)
    CCI = get_indicators('CCI',14,normed_prices,syms)

if __name__ == "__main__":
    test_code()