from activity_logger import activity_logger
from preprocessing import data_preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import operator
from pathlib import Path


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


class Model_Tuner:
    def __init__(self):
        self.activity_log_obj = activity_logger()
        
    def author(self):
        return 'Shrikanth Mahale'
    
    def lasso_feature_selection(self,X,y,file_object):
        try:
            self.activity_log_obj.log(file_object,'Checking for best feauture using Lasso')
            feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
            feature_sel_model.fit(X, y)
            feature_sel_model.get_support()
            selected_feat = X.columns[(feature_sel_model.get_support())]
            X = X[selected_feat]
            self.activity_log_obj.log(file_object,'select Best Features')
            return X
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception occured   ' + str(e))
            self.activity_log_obj.log(file_object,'Feature Selection Failed')
            raise Exception()
            
    def model_hyperparameter_tuning(self,X,y,file_object):
        try:
            model_params = {
                            'Decision_Tree': {
                                'model': DecisionTreeRegressor(),
                                'params': {
                                     "splitter"         : ["best","random"] ,
                                     "max_depth"        : [100,200,300,400,500],
                                     "min_samples_leaf" : [ 5,10,15,20,25],
                                     "min_weight_fraction_leaf":[0.01,0.1,1.0],
                                     "max_features"     : ["auto","log2","sqrt",None ],
                                     "max_leaf_nodes"   :[None,60,100,200,300]
                                }
                            },
                            'XG_Boost': {
                                'model': xgb.XGBRegressor(),
                                'params': { 'n_estimators': [44,60,80,100,120],
                                            'learning_rate': ['0.1'],
                                            'max_depth': [5],
                                            'subsample': [0.6],
                                            'min_child_weight': [int(x) for x in np.linspace(start = 1, stop = 100, num = 25)] }
                                }  
                        }
                
            scores = []
            for model_name, mp in model_params.items():
                print('Running ' + model_name)
                reg =  GridSearchCV(mp['model'], param_grid = mp['params'], cv=5, scoring='r2',n_jobs=-1,verbose=0)
                reg.fit(X, y)
                scores.append({
                    'model': model_name,
                    'best_score': reg.best_score_,
                    'best_params': reg.best_params_,
                    'best_estimators' : reg.best_estimator_
                })
                print(scores)
            hyperparameter_tuning_results = pd.DataFrame(scores,columns=['model','best_score','best_params','best_estimators'])
            return hyperparameter_tuning_results
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception occured   ' + str(e))
            self.activity_log_obj.log(file_object,'Model Tuning Failed')
            raise Exception()
            
            
    def get_best_estimator(self,X_train,y_train,X_test,y_test,file_object,hyperparameter_tuning_results):
        try:
            self.activity_log_obj.log(file_object,'Checking for the best model')
            hyperparameter_tuning_results.sort_values(by=['best_score'], ascending=False, inplace = True)
            hyperparameter_tuning_results = hyperparameter_tuning_results.reset_index(drop = True)
            X_train = self.lasso_feature_selection(X_train,y_train, file_object)
            X_test  = self.lasso_feature_selection(X_test,y_test, file_object)
            model_scores = dict()
            model_reg = dict()
            for i in range(0, len(hyperparameter_tuning_results)):
                model  = hyperparameter_tuning_results.loc[i,:]['model']
                regressor  = hyperparameter_tuning_results.loc[i,:]['best_estimators']
                self.activity_log_obj.log(file_object,'Testing the model {}'.format(model))
                reg = regressor
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                score = r2_score(y_test, y_pred)
                model_scores[model] = score
                model_reg[model] = regressor
                best_model = max(model_scores.items(), key=operator.itemgetter(1))[0]
                self.activity_log_obj.log(file_object,'The best model is {}'.format(best_model))
            return best_model,model_reg[best_model]
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception occured in the function get_best_estimator')
            raise Exception()
        
        
if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath("training_validation_insertion"))
    data = pd.read_csv(file_path +'\\Data_Training\\input.csv')

    Image_Folder = file_path +'\\Images\\' 
    Path(Image_Folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(Image_Folder))
    
    file_object = open("Training_Logs/Activity_Log.txt", 'a+')
    unwanted_columns = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year']
    
    data_preprocessing_obj = data_preprocessing()
    Model_Tuner_obj = Model_Tuner()
    
    final_data = data_preprocessing_obj.feature_engineering(file_object,data, Image_Folder)
    X,y = data_preprocessing_obj.feature_scaling_splitting(file_object,final_data,'VISIBILITY','NoScale')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)
    hyperparameter_tuning_results = Model_Tuner_obj.model_hyperparameter_tuning(X_train,y_train,file_object)
    best_model,best_estimator =  Model_Tuner_obj.get_best_estimator(X_train,y_train,X_test,y_test,file_object,hyperparameter_tuning_results)
    file_object.close()
    


