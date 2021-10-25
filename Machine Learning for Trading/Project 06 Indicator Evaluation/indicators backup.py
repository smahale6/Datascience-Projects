import pandas as pd
import numpy as np
import datetime as dt
import util
import matplotlib.pyplot as plt


def get_indicators(indicator, look_back_window , prices, sma_bbp = None):
    if indicator == 'SMA':
        sma = prices.rolling(window = look_back_window, center=False).mean()
        psma = sma / prices
        title="Price/SMA Ratio"
        sma_label = "SMA"
        psma_label = "Price/SMA"
        normed_prices_label = "Normalized Prices"
        Figure1, ax = plt.subplots()
        ax.set(xlabel='Time', ylabel = "Price", title=title)
        ax.plot(prices, label = normed_prices_label)
        ax.plot(sma, label = sma_label)
        ax.plot(psma, label = psma_label)
        ax.legend()
        Figure1.savefig('PriceSMA.png')
        plt.clf()
        return sma, psma
    elif indicator == 'Bollinger':
        rolling_std = prices.rolling(window = look_back_window, min_periods = look_back_window).std()
        top_bb = sma_bbp + (2 * rolling_std)
        bottom_bb = sma_bbp - (2 * rolling_std)
        Bollinger = (prices - bottom_bb) / (top_bb - bottom_bb)
        title = "Bollinger Bands"
        normed_prices_label = "Normalized Prices"
        UB_label = "Upper Band"
        LB_label = "Lower Band"
        Figure2, ax = plt.subplots()
        ax.set(xlabel='Time', ylabel="Price", title = title)
        ax.plot(prices, label = normed_prices_label)
        ax.plot(top_bb, label = UB_label)
        ax.plot(bottom_bb, label = LB_label)
        ax.legend()
        Figure2.savefig('Bollinger.png')
        plt.clf()
        return Bollinger
    elif indicator == 'Momentum':
        Momentum=(prices/prices.shift(look_back_window))-1
        Figure3, ax = plt.subplots()
        title = "Momentum"
        normed_prices_label = "Normalized Prices"
        ax.set(xlabel='Time', ylabel = "Price", title = title)
        ax.plot(prices, label = normed_prices_label)
        ax.plot(Momentum, label = "Momentum")
        ax.legend()
        Figure3.savefig('Momentum.png')
        plt.clf()
        return Momentum
    elif indicator == 'Volatility':
        DailyReturns = (prices/prices.shift(1)) - 1
        Volatility = DailyReturns.rolling(window = look_back_window, min_periods = look_back_window).std()
        Figure4, ax = plt.subplots()
        title = "Volatility"
        normed_prices_label = "Normalized Prices"
        ax.set(xlabel='Time', ylabel = "Price", title = title)
        ax.plot(prices, label = normed_prices_label)
        ax.plot(Volatility, label = "Volatility")
        ax.legend()
        Figure4.savefig('Volatility.png')
        plt.clf()
        return Volatility
    elif indicator == 'CCI':
        rm = prices.rolling(window=look_back_window,center=False).mean()
        cci = (prices-rm)/(2.5 * prices.std())
        Figure5, ax = plt.subplots()
        title = "Commodity Channel Index"
        normed_prices_label = "Normalized Prices"
        ax.set(xlabel='Time', ylabel = "Price", title = title)
        ax.plot(prices, label = normed_prices_label)
        ax.plot(cci, label = title)
        ax.legend()
        Figure5.savefig('Commodity Channel index.png')
        plt.clf()
        return Volatility


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
    
    SMA, PSMA = get_indicators('SMA',14,normed_prices)
    Bollinger = get_indicators('Bollinger',14,normed_prices,SMA)
    Momentum = get_indicators('Momentum',14,normed_prices)
    Volatility = get_indicators('Volatility',14,normed_prices)
    Volatility = get_indicators('CCI',14,normed_prices)

if __name__ == "__main__":
    test_code()