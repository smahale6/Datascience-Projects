import numpy as np
import operator
from operator import itemgetter
import pandas as pd
import random
from scipy import stats

class RTLearner(object):


    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.dt_tree = None
        if verbose == True:
            self.fetch_learner_info()

    def author():
        return 'smahale6'
        #Student Name: Shrikanth Mahale 		  	   		     		  		  		    	 		 		   		 		  
        #GT User ID: smahale6 	  	   		     		  		  		    	 		 		   		 		  
        #GT ID: 903453344  

    def build_dt_tree(self, trainX, trainY):
        if trainX.shape[0] <= self.leaf_size:
            return np.array([-1, trainY.mean(), np.nan, np.nan])
        if len(np.unique(trainY)) == 1:
            return np.array([-1, trainY.mean(), np.nan, np.nan])
        else:
            Best_Feature = random.randint(0, trainX.shape[1] - 1)
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
    
        
    def add_evidence(self, trainX, trainY):
        tree = self.build_dt_tree(trainX, trainY)
        if self.dt_tree == None:
            self.dt_tree = tree
        else:
            self.dt_tree = np.vstack((self.dt_tree, tree))
            


    def Find_dt_tree(self, test, row):
        feature, splitval = self.dt_tree[row, 0:2]
        if feature == -1:
            return splitval
        elif test[int(feature)] <= splitval:
            prediction = self.Find_dt_tree(test, row + int(self.dt_tree[row, 2]))
        else:
            prediction = self.Find_dt_tree(test, row + int(self.dt_tree[row, 3]))
        return prediction
        
    
    def query(self, TestX):
        predictions = list()
        for x in TestX:
            predictions.append(self.Find_dt_tree(x, row=0))
        return np.asarray(predictions)    
    
    def fetch_learner_info(self):
        print("Leaf Size =", self.leaf_size)
    
