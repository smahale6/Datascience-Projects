from activity_logger import activity_logger
from preprocessing import data_preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import operator

from sklearn import svm
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


class Model_Tuner:
    def __init__(self):
        self.activity_log_obj = activity_logger()
        
    def author(self):
        return 'Shrikanth Mahale'
        
    def model_hyperparameter_tuning(self,X,y,file_object):
        try:
            model_params = {
                # 'KNN': {
                #     'model': KNeighborsClassifier(),
                #     'params' : {
                #         'leaf_size': [10,20,30,40,50],
                #         'n_neighbors' : list(range(1,30)),
                #         'p': [1,2]
                #     }  
                # }, 
                'Logistic_Regression' : {
                    'model': LogisticRegression(solver='liblinear'),
                    'params': {
                        'C': list(range(1,10))
                    }
                },  
                'Random_Forest': {
                    'model': RandomForestClassifier(),
                    'params' : {
                        'n_estimators': list(range(1,50))
                    }
                },
                'Decision_Tree': {
                    'model': DecisionTreeClassifier(),
                    'params': {
                        'criterion': ['gini','entropy'],  
                        'splitter' : ['best','random'],
                        'min_samples_leaf' : list(range(1,50))
                    }
                },
                'SVM': {
                    'model': svm.SVC(gamma='auto'),
                    'params' : {
                        'C': list(range(1,10)),
                        'kernel': ['rbf','linear']
                    }  
                },
                'XG_Boost': {
                    'model': XGBClassifier(silent=True, verbosity = 0),
                    'params': { 'n_estimators' :  [10, 50, 100, 130], 
                                'criterion': ['gini', 'entropy'], 
                                'max_depth': range(2, 10, 1) }
                    }  
            }
            scores = []

            for model_name, mp in model_params.items():
                print('Running ' + model_name)
                clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
                clf.fit(X, y)
                scores.append({
                    'model': model_name,
                    'best_score': clf.best_score_,
                    'best_params': clf.best_params_,
                    'best_estimators' : clf.best_estimator_
                })
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
            model_scores = dict()
            model_clf = dict()
            for i in range(0, len(hyperparameter_tuning_results)):
                model  = hyperparameter_tuning_results.loc[i,:]['model']
                classifier  = hyperparameter_tuning_results.loc[i,:]['best_estimators']
                self.activity_log_obj.log(file_object,'Testing the model {}'.format(model))
                clf = classifier
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                if len(y_test.unique()) == 1:
                    score = accuracy_score(y_test, y_pred)
                else:
                    score = roc_auc_score(y_test, y_pred) 
                model_scores[model] = score
                model_clf[model] = classifier
                best_model = max(model_scores.items(), key=operator.itemgetter(1))[0]
                self.activity_log_obj.log(file_object,'The best model is {}'.format(best_model))
            return best_model,model_clf[best_model]
        except Exception as e:
            self.activity_log_obj.log(self.file_object,'Exception occuredin the function get_best_estimator')
            raise Exception()
        
        
if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath("training_validation_insertion"))
    data=pd.read_csv(file_path +'\\Data\\input.csv')

    
    file_object = open("Training_Logs/Activity_Log.txt", 'a+')
    unwanted_columns = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year']
    
    data_preprocessing_obj = data_preprocessing()
    Model_Tuner_obj = Model_Tuner()
    
    final_data,num_cols = data_preprocessing_obj.feature_engineering(file_object,data, unwanted_columns)
    mixed_minmax_cols = ['insured_education_level','incident_severity','property_damage'] 
    mixed_std_cols = num_cols
    X,y = data_preprocessing_obj.feature_scaling_splitting(file_object,final_data,'fraud_reported','Mixed',mixed_minmax_cols , mixed_std_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)
    hyperparameter_tuning_results = Model_Tuner_obj.model_hyperparameter_tuning(X_train,y_train,file_object)
    best_model,best_estimator =  Model_Tuner_obj.get_best_estimator(X_train,y_train,X_test,y_test,file_object,hyperparameter_tuning_results)
    file_object.close()
    
    