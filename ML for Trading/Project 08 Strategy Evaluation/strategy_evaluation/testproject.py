import pandas as pd
import numpy as np
import datetime as dt
import util
import matplotlib.pyplot as plt
import indicators
import ManualStrategy
import StrategyLearner
import experiment1
import experiment2


def author():
    return 'smahale6'
    #Student Name: Shrikanth Mahale 		  	   		     		  		  		    	 		 		   		 		  
    #GT User ID: smahale6 	  	   		     		  		  		    	 		 		   		 		  
    #GT ID: 903453344  

if __name__ == "__main__":
    symbol = 'JPM'
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sdo = dt.datetime(2010,1,1)
    edo = dt.datetime(2011,12,31)
    impact_values = [0.0,0.025,0.045,0.065,0.085,0.1]

    ####Execute Manual Strategy
    print('######In-Sample Stats##########')
    ManualStrategy.execution(symbol,sv,sd,ed,'In-Sample')
    print('######Out-of-Sample Stats##########')
    ManualStrategy.execution(symbol,sv,sdo,edo,'Out-Of-Sample')
    ####Execute Experiment 1
    experiment1.execution(symbol,sv,sd,ed)
    ####Execute Experiment 2
    experiment2.execution(symbol,sv,sd,ed,impact_values)