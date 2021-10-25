import datetime as dt
from util import get_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import marketsimcode
#from marketsimcode import compute_portvals


def author():
    return 'smahale6'
    #Student Name: Shrikanth Mahale 		  	   		     		  		  		    	 		 		   		 		  
    #GT User ID: smahale6 	  	   		     		  		  		    	 		 		   		 		  
    #GT ID: 903453344  

def compute_portvals( orders, start_val = 1000000, commission=9.95, impact=0.005, symbol = 'JPM'):
    orders.sort_index(ascending=True, inplace=True)
    start_date = orders.index.min()
    end_date = orders.index.max()
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices.fillna(method="ffill", inplace=True)  #Forward fill of null values
    prices.fillna(method="bfill", inplace=True)  #Backward fill of null values  
    prices.drop('SPY', 1, inplace= True)
    prices['CASH'] = 1.0
    Stock_Transaction = prices.copy()
    Stock_Transaction[:] = 0

    def get_stocks(date, shares):
        tic_sym = symbol
        netholdings = Stock_Transaction[tic_sym].sum()
        if shares == 1000 and netholdings == -1000:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] += 2000
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] -= (prices.loc[prices.index== date, tic_sym] * 2000 * (1 + impact)) + commission
        elif shares == 1000 and netholdings == 0:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] += 1000
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] -= (prices.loc[prices.index== date, tic_sym] * 1000 * (1 + impact)) + commission
        elif shares == 1000 and netholdings == 1000:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] += 0
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] -= (prices.loc[prices.index== date, tic_sym] * 0 * (1 + impact)) + commission       
        elif shares == -1000 and netholdings == -1000:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] -= 0
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] += (prices.loc[prices.index== date, tic_sym] * 0 * (1 + impact)) + commission
        elif shares == -1000 and netholdings == 0:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] -= 1000
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] += (prices.loc[prices.index== date, tic_sym] * 1000 * (1 + impact)) + commission
        elif shares == -1000 and netholdings == 1000:
            Stock_Transaction.loc[Stock_Transaction.index== date, tic_sym] -= 2000
            Stock_Transaction.loc[Stock_Transaction.index== date, 'CASH'] += (prices.loc[prices.index== date, tic_sym] * 2000 * (1 + impact)) + commission
        
    orders['Date'] = orders.index    
    orders.apply(lambda x: get_stocks(x['Date'], x['Shares']), axis =1)
    
    holdings = Stock_Transaction.copy()
    holdings.loc[holdings.index == start_date, 'CASH']+= start_val
    holdings = holdings.cumsum()
    value = prices * holdings
    portval = value.sum(axis=1)
    
    dailyReturns = portval.copy()
    dailyReturns[1:] = (portval[1:] / portval[:-1].values) - 1
    dailyReturns.iloc[0] = 0
    dailyReturns = dailyReturns[1:]
    cr = (portval[-1] / portval[0]) - 1
    adr = dailyReturns.mean()
    sddr = dailyReturns.std()
    return portval, cr, adr, sddr

def testPolicy(symbol, start_date, end_date, sv = 100000):
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices.fillna(method="ffill", inplace=True)  #Forward fill of null values
    prices.fillna(method="bfill", inplace=True)  #Backward fill of null values
    prices = prices.loc[:,symbol]
    adjusted_prices = pd.Series(np.nan, index=prices.index)
    adjusted_prices[:-1] = prices[:-1] / prices.values[1:] - 1
    signs = (-1) * adjusted_prices.apply(np.sign)
    orders = signs.diff() / 2
    orders[0] = signs[0]
    trades = dict()
    total_orders = len(orders)
    order = 0 
    while order < total_orders:
        if orders[orders.index[order]] == 1:
            trades[orders.index[order]] = 1000
        elif orders[orders.index[order]] == -1:
            trades[orders.index[order]] = -1000     
        elif orders[orders.index[order]] == 0:
            trades[orders.index[order]] = 0
        order = order + 1
    df_trades = pd.DataFrame(list(trades.items()), columns=["Date", "Shares"])
    df_trades.set_index("Date", inplace=True)
    return df_trades

def portfolio_values(trades,sv,start_date,end_date,symbol):
    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices[symbol]
    prices_symbol_len = len(prices)
    benchmark_trades = np.zeros(prices_symbol_len)
    benchmark_trades[0] = 1000
    benchmark_trades = pd.DataFrame(data=benchmark_trades, index=prices.index, columns=['Shares'])
    benchmark_portvals, benchmark_cumulative_returns, benchmark_mean, benchmark_std = compute_portvals(benchmark_trades, sv, 0.0, 0.0)
    optimal_portvals, optimal_cumulative_returns, optimal_mean, optimal_std = compute_portvals(trades, sv, 0.0, 0.0)
    return benchmark_portvals,benchmark_cumulative_returns,benchmark_mean,benchmark_std,optimal_portvals, optimal_cumulative_returns, optimal_mean, optimal_std 

def execution():
    sv = 100000
    symbol = 'JPM'
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    trades = testPolicy(symbol,start_date,end_date)
    benchmark_portvals,benchmark_cumulative_returns,benchmark_mean,benchmark_std,optimal_portvals, optimal_cumulative_returns, optimal_mean, optimal_std  = portfolio_values(trades,sv,start_date,end_date,symbol)
    
    normalized_optimal = optimal_portvals/optimal_portvals.iloc[0]
    normalized_benchmark = benchmark_portvals/benchmark_portvals.iloc[0]
    
    plt.title("TOS vs. Benchmark")
    plt.xlabel("Dates")
    plt.ylabel("Normalized Value of Portfolio")
    plt.plot(normalized_optimal, 'red', label="Optimal")
    plt.plot(normalized_benchmark, 'green', label="Benchmark")
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig("TOS vs. Benchmark.png")
    plt.clf()
    
    print("Optimal Cumulative Returns: ", optimal_cumulative_returns)
    print("Optimal Average Daily Returns: ", optimal_mean)
    print("Optimal Standard Deviation: ", optimal_std)
    print("Benchmark Cumulative Returns: ", benchmark_cumulative_returns)
    print("Benchmark Average Daily Returns: ", benchmark_mean)
    print("Benchmark Standard Deviation: ", benchmark_std)

if __name__ == "__main__":
    execution()
    
