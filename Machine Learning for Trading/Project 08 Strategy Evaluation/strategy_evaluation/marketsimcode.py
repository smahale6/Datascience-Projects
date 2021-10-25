import pandas as pd
from util import get_data
import math
## Shrikanth Mahale
## smahale6

def author():
    return 'smahale6'
    #Student Name: Shrikanth Mahale 		  	   		     		  		  		    	 		 		   		 		  
    #GT User ID: smahale6 	  	   		     		  		  		    	 		 		   		 		  
    #GT ID: 903453344  

def compute_portvals(orders, start_val = 1000000, impact=0.005,commission=9.95,symbol = ["JPM"]):
    orders_copy = orders.copy()
    def get_trades(date, shares):
        tic_sym = symbol[0]   
        if shares > 0:
            trades.loc[trades.index== date, tic_sym] += shares
            trades.loc[trades.index== date, 'CASH'] -= (prices.loc[prices.index== date, tic_sym] * shares * (1 + impact)) + commission
        elif shares < 0:
            trades.loc[trades.index== date, tic_sym] += shares
            trades.loc[trades.index== date, 'CASH'] += (prices.loc[prices.index== date, tic_sym] * abs(shares) * (1 - impact)) - commission
    
    orders_copy.sort_index(ascending=True, inplace=True)
    start_date = orders_copy.index.min()
    end_date = orders_copy.index.max()
    dates = pd.date_range(start_date, end_date)  
    prices = get_data(symbol, dates)
    prices = prices[symbol]
    prices['CASH'] = 1.0
    trades = prices.copy()
    trades[:] = 0
    orders_copy['date'] = orders_copy.index
    orders_copy.apply(lambda x: get_trades(x['date'], x['JPM']), axis =1)
    holdings = trades.copy()
    holdings.loc[holdings.index == start_date, 'CASH']+= start_val
    holdings = holdings.cumsum()

    value = prices * holdings
    portval = value.sum(axis=1)
    DailyReturns = (portval/portval.shift(1)) - 1
    DailyReturns = DailyReturns[1:]
    cr = (portval[-1] / portval[0]) - 1
    adr = DailyReturns.mean()
    sddr = DailyReturns.std()
    Sharpe_Ratio = math.sqrt(252) * (DailyReturns.mean() / DailyReturns.std())

    return portval, cr, adr, sddr, Sharpe_Ratio

