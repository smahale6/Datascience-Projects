  	   		     		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		     		  		  		    	 		 		   		 		  
import pandas as pd  		  	   	
import math	     		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  
import scipy.optimize as spo	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
def optimize_portfolio(  		  	   		     		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		     		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		     		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		     		  		  		    	 		 		   		 		  
):  		  	   		     		  		  		    	 		 		   		 		  
	  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		     		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		     		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		

    prices_all.fillna(method="ffill", inplace=True)  #Forward fill of null values
    prices_all.fillna(method="bfill", inplace=True)  #Backward fill of null values

    prices = prices_all[syms]  # only portfolio symbols  


    normed_prices = prices / prices.iloc[0]	 # Normed = Prices / First row of prices

    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  
    normed_SPY = prices_SPY/prices_SPY.iloc[0]
    		     		  		  		    	 		 		   		 		  


    def Sharpe_Ratio(allocation_estimation, normed_prices):
        allocated = normed_prices * allocation_estimation
        port_val = allocated.sum(axis=1)
        port_val_copy = port_val.copy()
        DailyReturns = (port_val_copy/port_val_copy.shift(1)) - 1
        DailyReturns.iloc[0] = 0
        DailyReturns = DailyReturns[1:]
        Sharpe_Ratio = math.sqrt(252) * (DailyReturns.mean() / DailyReturns.std())
        return (Sharpe_Ratio * -1)

    allocation_estimation = np.asarray([1.0/len(syms)] * len(syms))
    bounds = [(0.0, 1.0)] * len(syms)
    constraints  = {'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)}
    optimizer = spo.minimize(Sharpe_Ratio, allocation_estimation,args=(normed_prices, ), method='SLSQP', options={'disp':True}, bounds=bounds, constraints=constraints )
    allocs = optimizer.x # allocations


    alloced = normed_prices * allocs
    port_val = alloced.sum(axis=1) 
    port_val_copy = port_val.copy()
    DailyReturns = (port_val_copy/port_val_copy.shift(1)) - 1
    DailyReturns.iloc[0] = 0
    DailyReturns = DailyReturns[1:]
    
    cr = (port_val[-1] / port_val[0]) - 1  ## cumulative return
    adr = DailyReturns.mean()  # Average Daily Returns
    sddr = DailyReturns.std()  # Volatility (stdev of daily returns)
    sr = Sharpe_Ratio(allocation_estimation,normed_prices) # Sharpe Ratio


    if gen_plot:  		   	  			    		  		  		    	 		 		   		 		  
        # add code to plot here  		   	  			    		  		  		    	 		 		   		 		  
        df_temp = pd.concat([port_val, normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.plot()
        plt.xlabel("Date")
        plt.ylabel("Nomalized Price")
        plt.title("Daily Portfolio Value and SPY")
        plt.grid()
        plt.legend()
        plt.savefig("OptimizedFigure.png")
    return allocs, cr, adr, sddr, sr  

        
  		  	   		     		  		  		    	 		 		   		 		  
def test_code():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		   	   		     		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 6, 1)  		  	   		     		  		  		    	 		 		   		 		  
    end_date = dt.datetime(2009, 6, 1)  		  	   		     		  		  		    	 		 		   		 		  
    symbols = ["IBM", "X", "GLD", "JPM"]  		  	   		     		  		  		    	 		 		   		 		  	   		     		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		     		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=True)  		  	   		     		  		  		    	 		 		   		 		  	   		     		  		  		    	 		 		   		 		  
    # Print statistics  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		     		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
    test_code()  		  	   		     		  		  		    	 		 		   		 		  
