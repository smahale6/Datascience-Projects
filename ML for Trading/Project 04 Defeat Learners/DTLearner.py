""""""  		  	   		     		  		  		    	 		 		   		 		  
"""  		  	   		     		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		     		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		     		  		  		    	 		 		   		 		  
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
  		  	   		     		  		  		    	 		 		   		 		  
import warnings  		  	   		     		  		  		    	 		 		   		 		  	     		  		  		    	 		 		   		 		  
import numpy as np  
import 	operator
from operator import itemgetter
import pandas as pd	  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
class DTLearner(object):  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with  		  	   		     		  		  		    	 		 		   		 		  
    your own correct DTLearner from Project 3.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.  		  	   		     		  		  		    	 		 		   		 		  
    :type leaf_size: int  		  	   		     		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		     		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		     		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		     		  		  		    	 		 		   		 		  
    """  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  		  	   		     		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size=1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.dt_tree = None
        if verbose == True:
            self.fetch_learner_info()	  	   		     		  		  		    	 		 		   		 		  	  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def author(self):  		  	   		     		  		  		    	 		 		   		 		    	   		     		  		  		    	 		 		   		 		  
        return "smahale6"  # replace tb34 with your Georgia Tech username  		 


    def build_dt_tree(self, trainX, trainY):
        if trainX.shape[0] <= self.leaf_size:
            return np.array([-1, trainY.mean(), np.nan, np.nan])
        if len(np.unique(trainY)) == 1:
            return np.array([-1, trainY.mean(), np.nan, np.nan])
        else:
            correlations = dict()
            feature = 0
            total_features = trainX.shape[1]
            while feature < total_features:
                corr = np.corrcoef(trainX[:, feature], trainY)
                absolutecorr = abs(corr[0,1])
                correlations[feature] = absolutecorr
                feature = feature + 1
            Best_Feature = max(correlations.items(), key=operator.itemgetter(1))[0]
            Best_Feature_Values = trainX[:, Best_Feature]
            SplitVal = np.median(Best_Feature_Values)
            if (np.all(Best_Feature_Values <= SplitVal)):
                return np.array([-1, trainY.mean(), np.nan, np.nan])
            Left_Tree = self.build_dt_tree(trainX[Best_Feature_Values<= SplitVal], trainY[Best_Feature_Values<= SplitVal])
            Right_Tree = self.build_dt_tree(trainX[Best_Feature_Values> SplitVal], trainY[Best_Feature_Values> SplitVal])
            if Left_Tree.ndim == 1:
                Right_Start_Index = 2
            elif Left_Tree.ndim > 1:
                Right_Start_Index = Left_Tree.shape[0] + 1
            root = np.array([Best_Feature, SplitVal, 1, Right_Start_Index])
            return np.vstack((root, Left_Tree, Right_Tree))	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
    def add_evidence(self, data_x, data_y):
        tree = self.build_dt_tree(data_x, data_y)
        if self.dt_tree == None:
            self.dt_tree = tree
        else:
            self.dt_tree = np.vstack((self.dt_tree, tree))		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Add training data to learner  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		     		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		     		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        # slap on 1s column so linear regression finds a constant term  		  	   		     		  		  		    	 		 		   		 		  
        new_data_x = np.ones([data_x.shape[0], data_x.shape[1] + 1])  		  	   		     		  		  		    	 		 		   		 		  
        new_data_x[:, 0 : data_x.shape[1]] = data_x  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        # build and save the model  		  	   		     		  		  		    	 		 		   		 		  
        self.model_coefs, residuals, rank, s = np.linalg.lstsq(  		  	   		     		  		  		    	 		 		   		 		  
            new_data_x, data_y, rcond=None  		  	   		     		  		  		    	 		 		   		 		  
        )  		  	   		     		  		  		    	 		 		   		 		  

    def tree_search(self, test, row):
        feature, SplitVal = self.dt_tree[row, 0:2]
        if feature == -1:
            return SplitVal
        elif test[int(feature)] <= SplitVal:
            prediction = self.tree_search(test, row + int(self.dt_tree[row, 2]))
        else:
            prediction = self.tree_search(test, row + int(self.dt_tree[row, 3]))
        return prediction
          		     		  		  		    	 		 		   		 		  
    def query(self, points):
        predictions = []
        for i in points:
            predictions.append(self.tree_search(i, row=0))
        return np.asarray(predictions)    
    		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        Estimate a set of test points given the model we built.  		  	   		     		  		  		    	 		 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
        :param points: A numpy array with each row corresponding to a specific query.  		  	   		     		  		  		    	 		 		   		 		  
        :type points: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        :return: The predicted result of the input data according to the trained model  		  	   		     		  		  		    	 		 		   		 		  
        :rtype: numpy.ndarray  		  	   		     		  		  		    	 		 		   		 		  
        """  		  	   		     		  		  		    	 		 		   		 		  
        return (self.model_coefs[:-1] * points).sum(axis=1) + self.model_coefs[  		  	   		     		  		  		    	 		 		   		 		  
            -1  		  	   		     		  		  		    	 		 		   		 		  
        ]  		  	   		     		  		  		    	 		 		   		 		  

    def fetch_learner_info(self):
        print("Leaf Size =", self.leaf_size)	
        print("model_coefs =", self.model_coefs)  		  		    	 	
        print("residuals =", self.residuals)  			 		   		 		  
  		  	   		     		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		     		  		  		    	 		 		   		 		  
    print("the secret clue is 'zzyzx'")  		  	   		     		  		  		    	 		 		   		 		  
