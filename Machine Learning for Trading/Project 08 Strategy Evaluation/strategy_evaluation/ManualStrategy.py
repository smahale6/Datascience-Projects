import datetime as dt
from util import get_data
import pandas as pd
import numpy as np
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from indicators import get_indicators
import warnings
warnings.filterwarnings('ignore')
import math

def author():
    return 'smahale6'
    #Student Name: Shrikanth Mahale 		  	   		     		  		  		    	 		 		   		 		  
    #GT User ID: smahale6 	  	   		     		  		  		    	 		 		   		 		  
    #GT ID: 903453344  

def get_trades(orders,symbol, PSMA, BBP, volatility):
    trades = dict()
    holdings = 0.0
    index = 0
    order_index = list(orders.index)
    order_index_len = len(order_index)
    purchase_date = list()
    sell_date = list()
    while index < order_index_len:
        trade = 0
        if PSMA.iloc[index,0] <1 and BBP.iloc[index, 0]<0 and volatility.iloc[index,0] < 0.095:
            orders.loc[order_index[index],symbol] = 1
            trade = 1000 - holdings
            trades[order_index[index]] = trade
            purchase_date.append(order_index[index])
        elif PSMA.iloc[index,0] > 1.05 and BBP.iloc[index, 0]>1 and volatility.iloc[index,0] > 0.01:
            orders.loc[order_index[index],symbol]= -1
            trade = -1000 - holdings
            trades[order_index[index]] = trade
            sell_date.append(order_index[index])
        else:
            orders.loc[order_index[index],symbol] = 0
            trade = 0
            trades[order_index[index]] = trade
        holdings = holdings + trade
        index += 1
    return trades, purchase_date, sell_date

def testPolicy(symbol, sd, ed, sv = 100000):
    prices = get_data([symbol], pd.date_range(sd, ed))
    prices.fillna(method="ffill", inplace=True)  #Forward fill of null values
    prices.fillna(method="bfill", inplace=True)  #Backward fill of null values
    prices = prices[[symbol]]
    SMA,PSMA = get_indicators('SMA',14,prices,[symbol])
    BBP = get_indicators('Bollinger',14,prices,[symbol])
    volatility = get_indicators('Volatility',14,prices,symbol)
    orders = prices.copy()
    orders[:] = 0
    trades, purchase_date, sell_date = get_trades(orders,symbol, PSMA, BBP, volatility)
    df_trades = pd.DataFrame(list(trades.items()), columns=["Date", symbol])    
    df_trades.set_index("Date", inplace=True)
    return df_trades,purchase_date, sell_date 

def portfolio_values(trades,sv,sd,ed,symbol,impact= 0.005, commission = 9.95):
    prices = get_data([symbol], pd.date_range(sd, ed))
    prices.fillna(method="ffill", inplace=True)  #Forward fill of null values
    prices.fillna(method="bfill", inplace=True)  #Backward fill of null values
    prices = prices[symbol]
    prices_symbol_len = len(prices)
    benchmark_trades = np.zeros(prices_symbol_len)
    benchmark_trades[0] = 1000
    benchmark_trades = pd.DataFrame(data=benchmark_trades, index=prices.index, columns=[symbol])
    benchmark_portvals, benchmark_cumulative_returns, benchmark_mean, benchmark_std, benchmark_Sharpe_Ratio = compute_portvals(benchmark_trades, sv,impact, commission,[symbol])
    optimal_portvals, optimal_cumulative_returns, optimal_mean, optimal_std,optimal_Sharpe_Ratio = compute_portvals(trades, sv,  impact, commission,[symbol])
    return benchmark_portvals,benchmark_cumulative_returns,benchmark_mean,benchmark_std,benchmark_Sharpe_Ratio ,optimal_portvals, optimal_cumulative_returns, optimal_mean, optimal_std,optimal_Sharpe_Ratio 
  


def execution(symbol,sv,sd,ed,data_category):
    
    assert data_category == 'In-Sample' or data_category == 'Out-Of-Sample'
    
    trades,purchase_date, sell_date  = testPolicy(symbol,sd,ed,sv)
    benchmark_portvals,benchmark_cumulative_returns,benchmark_mean,benchmark_std,benchmark_Sharpe_Ratio ,optimal_portvals, optimal_cumulative_returns, optimal_mean, optimal_std,optimal_Sharpe_Ratio  = portfolio_values(trades,sv,sd, ed,symbol,impact= 0.005, commission = 9.95)
    
    normalized_optimal = optimal_portvals/optimal_portvals.iloc[0]
    normalized_benchmark = benchmark_portvals/benchmark_portvals.iloc[0]
    

    print("Manual Cumulative Returns: ", optimal_cumulative_returns)
    print("Manual Average Daily Returns: ", optimal_mean)
    print("Manual Standard Deviation: ", optimal_std)
    print("Manual Sharpe Ratio: ", optimal_Sharpe_Ratio)
    print("Benchmark Cumulative Returns: ", benchmark_cumulative_returns)
    print("Benchmark Average Daily Returns: ", benchmark_mean)
    print("Benchmark Standard Deviation: ", benchmark_std)
    print("Benchmark Sharpe Ratio: ", benchmark_Sharpe_Ratio)
    
    plt.title("Manual Strategy vs. Benchmark Strategy - " + data_category)
    plt.ylabel("Normalized Portfolio Value")
    plt.xlabel("Dates")
    plt.xticks(rotation=45)
    plt.plot(normalized_optimal, 'r', label="Manual Strategy")
    plt.plot(normalized_benchmark, 'g', label="Benchmark Strategy")
    for date in purchase_date:
        plt.axvline(date,color="blue", linewidth = 0.5)
    for date in sell_date:
        plt.axvline(date,color="black", linewidth = 0.5)
    plt.legend()
    plt.savefig("ManualStrategy - " + data_category + ".png")
    plt.show()
    plt.clf()



if __name__ == "__main__":
    sv = 100000
    symbol = 'JPM'
    
    print('######In-Sample Stats##########')
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    execution(symbol,sv,sd,ed,'In-Sample')
    
    print('######Out-of-Sample Stats##########')
    sdo = dt.datetime(2010,1,1)
    edo = dt.datetime(2011,12,31)
    execution(symbol,sv,sdo,edo,'Out-Of-Sample')










