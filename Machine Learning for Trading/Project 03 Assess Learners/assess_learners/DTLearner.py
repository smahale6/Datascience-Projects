import numpy as np
import operator
from operator import itemgetter
import pandas as pd

class DTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.dt_tree = None

    def author(self):
        return 'Shrikanth Mahale'

    def build_dt_tree(self, xTrain, yTrain):
        if xTrain.shape[0] <= self.leaf_size:
            return np.array([-1, yTrain.mean(), np.nan, np.nan])
        if len(np.unique(yTrain)) == 1:
            return np.array([-1, yTrain.mean(), np.nan, np.nan])
        else:
            correlations = dict()
            feature = 0
            total_features = xTrain.shape[1]
            while feature < total_features:
                corr = np.corrcoef(xTrain[:, feature], yTrain)
                absolutecorr = abs(corr[0,1])
                correlations[feature] = absolutecorr
                feature = feature + 1
            Best_Feature = max(correlations.items(), key=operator.itemgetter(1))[0]
            Best_Feature_Values = xTrain[:, Best_Feature]
            SplitVal = np.median(Best_Feature_Values)
            if (np.all(Best_Feature_Values <= SplitVal)):
                return np.array([-1, yTrain.mean(), np.nan, np.nan])
            Left_Tree = self.build_dt_tree(xTrain[Best_Feature_Values<= SplitVal], yTrain[Best_Feature_Values<= SplitVal])
            Right_Tree = self.build_dt_tree(xTrain[Best_Feature_Values> SplitVal], yTrain[Best_Feature_Values> SplitVal])
            if Left_Tree.ndim == 1:
                Right_Start_Index = 2
            elif Left_Tree.ndim > 1:
                Right_Start_Index = Left_Tree.shape[0] + 1
            root = np.array([Best_Feature, SplitVal, 1, Right_Start_Index])
            return np.vstack((root, Left_Tree, Right_Tree))
        
    def add_evidence(self, xTrain, yTrain):
        tree = self.build_dt_tree(xTrain, yTrain)
        if self.dt_tree == None:
            self.dt_tree = tree
        else:
            self.dt_tree = np.vstack((self.dt_tree, tree))
            
    def tree_search(self, test, row):
        feature, SplitVal = self.dt_tree[row, 0:2]
        if feature == -1:
            return SplitVal
        elif test[int(feature)] <= SplitVal:
            prediction = self.tree_search(test, row + int(self.dt_tree[row, 2]))
        else:
            prediction = self.tree_search(test, row + int(self.dt_tree[row, 3]))
        return prediction
    
    def query(self, xTest):
        predictions = []
        for i in xTest:
            predictions.append(self.tree_search(i, row=0))
        return np.asarray(predictions)    
    
    
    
