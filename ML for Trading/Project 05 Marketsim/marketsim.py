""""""  		  	   		     		  		  		    	 		 		   		 		  
"""MC2-P1: Market simulator.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		     		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  
import os  		  	   		     		  		  		    	 		 		   		 		  		  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
import pandas as pd
import  util 		  	   		     		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  
import math	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def author():
    return 'Shrikanth Mahale'

def compute_portvals(  		  	   		     		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		     		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		     		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		     		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		     		  		  		    	 		 		   		 		  
):  		  	   		     		  		  		    	 		 		   		 		  


    def get_trades(tic_sym, date, shares, order):
        if order == "BUY":
            trades.loc[trades.index== date, tic_sym]+= shares
            trades.loc[trades.index== date, 'CASH']-= (prices.loc[prices.index== date, tic_sym] * shares * (1 + impact)) + commission
        elif order == "SELL":
            trades.loc[trades.index== date, tic_sym] -= shares
            trades.loc[trades.index== date, 'CASH'] += (prices.loc[prices.index== date, tic_sym] * shares * (1 - impact)) - commission

    orders = pd.read_csv(orders_file)
    start_date = orders.loc[:,'Date'].min()
    end_date = orders.loc[:,'Date'].max()
    dates = pd.date_range(start_date, end_date) 

    symbols = list(orders.loc[:, 'Symbol'].unique())

    prices = get_data(symbols, dates)
    prices.fillna(method="ffill", inplace=True)  #Forward fill of null values
    prices.fillna(method="bfill", inplace=True)  #Backward fill of null values
    prices = prices[symbols]
    prices['CASH'] = 1.0
    trades = prices.copy()
    trades[:] = 0

    orders.apply(lambda x: get_trades(x['Symbol'], x['Date'], x['Shares'], x['Order']), axis =1)
    
    holdings = trades.copy()
    holdings.loc[holdings.index == start_date, 'CASH']+= start_val
    holdings = holdings.cumsum()
    
    value = prices * holdings
    portval = value.sum(axis=1)

    return portval		  	   		     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def test_code():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		     		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		     		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    of = "./orders/orders-09.csv"	  	   		     		  		  		    	 		 		   		 		  
    sv = 1000000  		  	   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # Process orders  		  	   		     		  		  		    	 		 		   		 		  

    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	
       		     		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		     		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		     		  		  		    	 		 		   		 		  
    else:  		  	   		     		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  			   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # Get portfolio stats  		   	  			    		  		  		    	 		 		   		 		  
    # Here we just fake the data. you should use your code from previous assignments.  		   	  			    		  		  		    	 		 		   		 		  
    start_date = portvals.index.min()##dt.datetime(2011,1,10)  		  	   		     		  		  		    	 		 		   		 		  
    end_date = portvals.index.max() ##dt.datetime(2011,3,3) 	

    port_val = portvals[(portvals.index >= start_date) & (portvals.index <= end_date)]
    DailyReturns = (port_val/port_val.shift(1)) - 1

    cum_ret = (port_val[-1] / port_val[0]) - 1  ## cumulative return
    avg_daily_ret = DailyReturns.mean()  # Average Daily Returns
    std_daily_ret = DailyReturns.std()  # Volatility (stdev of daily returns)
    sharpe_ratio = math.sqrt(252) * (DailyReturns.mean() / DailyReturns.std())  	   	
    	     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5,] 		  	   		     		  		  		    	 		 		   		 		  
  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
    # Compare portfolio against $SPX  		   	  			    		  		  		    	 		 		   		 		  
    print("Date Range: {} to {}".format(start_date, end_date)) 		   	  			    		  		  		    	 		 		   		 		  
    print("") 		   	  			    		  		  		    	 		 		   		 		  
    print("Sharpe Ratio of Fund: {}".format(sharpe_ratio))  		   	  			    		  		  		    	 		 		   		 		  
    print("Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY))  		   	  			    		  		  		    	 		 		   		 		  
    print("") 		   			    		  		  		    	 		 		   		 		  
    print("Cumulative Return of Fund: {}".format(cum_ret))  		   	  			    		  		  		    	 		 		   		 		  
    print("Cumulative Return of SPY : {}".format(cum_ret_SPY))  		   	  			    		  		  		    	 		 		   		 		  
    print("") 		   		   	  			    		  		  		    	 		 		   		 		  
    print("Standard Deviation of Fund: {}".format(std_daily_ret))  		   	  			    		  		  		    	 		 		   		 		  
    print("Standard Deviation of SPY : {}".format(std_daily_ret_SPY))  		   	  			    		  		  		    	 		 		   		 		  
    print("") 		   		   	  			    		  		  		    	 		 		   		 		  
    print("Average Daily Return of Fund: {}".format(avg_daily_ret))  		   	  			    		  		  		    	 		 		   		 		  
    print("Average Daily Return of SPY : {}".format(avg_daily_ret_SPY))  		   	  			    		  		  		    	 		 		   		 		  
    print("") 		    		   	  			    		  		  		    	 		 		   		 		  
    print("Final Portfolio Value: {}".format(portvals[-1]))		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == '__main__':
    test_code()  		  	   		     		  		  		    	 		 		   		 		  
