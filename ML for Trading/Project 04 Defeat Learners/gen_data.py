""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		     		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		     		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		     		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		     		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		     		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		     		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		     		  		  		    	 		 		   		 		  
or edited.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		     		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		     		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		     		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
Student Name: Shrikanth Mahale (replace with your name)  		  	   		     		  		  		    	 		 		   		 		  
GT User ID: smahale6 (replace with your User ID)  		  	   		     		  		  		    	 		 		   		 		  
GT ID: 903453344 (replace with your GT ID)  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
import math  		  	   		     		  		  		    	 		 		   		 		     		     		  		  		    	 		 		   		 		  
import numpy as np  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
# this function should return a dataset (X and Y) that will work  		  	   		     		  		  		    	 		 		   		 		  
# better for linear regression than decision trees  		  	   		     		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1489683273):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		     		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		     		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		     		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		     		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    np.random.seed(seed)  		  	   		     		  		  		    	 		 		   		 		  
    rows = np.random.randint(10, 1001)
    columns = np.random.randint(2, 11)
    x = np.random.rand(rows, columns)		  	   		     		  		  		    	 		 		   		 		  
    y = np.zeros(rows)		  
    coeff = np.random.rand(columns,1).tolist()     		  		  		    	 		 		   		 		  
    # Here's is an example of creating a Y from randomly generated  		  	   		     		  		  		    	 		 		   		 		  
    # X with multiple columns  		  	   		     		  		  		    	 		 		   		 		  
    # y = x[:,0] + np.sin(x[:,1]) + x[:,2]**2 + x[:,3]**3  	
    column  = 0	  	   		     		  		  		    	 		 		   		 		  
    while column < columns:
        y = y[column] + (coeff[column]*x[:, column])
        column += 1
    return x, y  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def best_4_dt(seed=1489683273):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		     		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		     		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		     		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		     		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    np.random.seed(seed)
    rows = np.random.randint(10, 1001)
    columns = np.random.randint(2, 11)
    x = np.random.rand(rows, columns)   		     		  		  		    	 		 		   		 		  
    y = np.zeros(rows)	
    feature = np.random.randint(0, columns) 	
    feature_median = np.median(x[:,feature])	  
    lower_leaf = np.random.randint(0, 100)
    upper_leaf = np.random.randint(100, 200)	   		     		  		  		    	 		 		   		 		  
    row  = 0	  	   		     		  		  		    	 		 		   		 		  
    while row < rows:
        if x[row, feature] < feature_median:
            y[row] = lower_leaf
        else:
            y[row] = upper_leaf
        row += 1
    return x, y  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
def author():  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		     		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    return "smahale6"  # Change this to your user ID  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    print("they call me Tim.")  		  	   		     		  		  		    	 		 		   		 		  
