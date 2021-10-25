import pandas as pd
import numpy as np
import datetime as dt
import util
import matplotlib.pyplot as plt
import indicators
import TheoreticallyOptimalStrategy


def author():
    return 'smahale6'
    #Student Name: Shrikanth Mahale 		  	   		     		  		  		    	 		 		   		 		  
    #GT User ID: smahale6 	  	   		     		  		  		    	 		 		   		 		  
    #GT ID: 903453344  

if __name__ == "__main__":
    ##Executing Indicators Output
    indicators.execute_indicators()
    plt.clf()
    ##Executing TheoreticallyOptimalStrategy output
    TheoreticallyOptimalStrategy.execution()

    