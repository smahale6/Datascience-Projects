import os
from activity_logger import activity_logger
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

class data_preprocessing:
    def __init__(self):
        self.activity_log_obj = activity_logger()
        
    def author(self):
        return 'Shrikanth Mahale'
        
    def removing_unwanted_space(self,file_object,data):
        try:
            ## removing unnecessary spaces for categorical variables
            self.activity_log_obj.log(file_object, 'removing unnecessary spaces for categorical variables')
            df_changed=data.apply(lambda x: x.str.strip() if x.dtype == "object" else x) 
            self.activity_log_obj.log(file_object, 'removed unwanted spaces for categorical variables')
            return df_changed
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception Occured. Exception message:  ' + str( e))
            self.activity_log_obj.log(file_object,'Unwanted space/column removal Unsuccessful. Exited the remove_unwanted_spaces method of the Preprocessor class')
            raise Exception()
    
    
    
    def removing_unwanted_columns(self,file_object,data,unwanted_columns):
        try:
            ## removing unwanted columns for categorical variables
            self.activity_log_obj.log(file_object, 'Removing unwanted columns')
            data.drop(columns=unwanted_columns,inplace=True)
            self.activity_log_obj.log(file_object, 'Removed unwanted columns. Columns removed are {}'.format(','.join(unwanted_columns)))
            return data
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception Occured. Exception message:  ' + str( e))
            self.activity_log_obj.log(file_object,'Unwanted space/column removal Unsuccessful. Exited the remove_unwanted_spaces method of the Preprocessor class')
            raise Exception()
    
    def feature_scaling_splitting(self,file_object, data,response_column,scaling_type, mixed_minmax_cols = [], mixed_std_cols = []):
        try:
            ## Splitting datasets into predictors and response
            self.activity_log_obj.log(file_object, 'Splitting Predictors and response')
            if response_column != None:
                X = data.drop(response_column,axis=1)
                y = data[response_column]
            else:
                X = data
            self.activity_log_obj.log(file_object, 'Predictors and Response have been split')
            
            MinMaxTransformer = MinMaxScaler()
            StandardTransformer = StandardScaler()
            if scaling_type not in ['MinMax','Standard','Mixed']:
                raise ValueError("Only allowed Values are 'MinMax','Standard' and 'Mixed'")
            elif scaling_type == 'MinMax':
                X_Scaled = MinMaxTransformer.fit_transform(X)
                X_Scaled = pd.DataFrame(data=X_Scaled, columns=X.columns,index=X.index)
                self.activity_log_obj.log(file_object, 'Performed MinMax Scaling on all columns')
            elif scaling_type == 'Standard':
                X_Scaled = StandardScaler.fit_transform(X)
                X_Scaled = pd.DataFrame(data=X_Scaled, columns=X.columns,index=X.index)
                self.activity_log_obj.log(file_object, 'Performed Standard Scaling on all columns')
            else:
                preprocessor = ColumnTransformer(
                    remainder='passthrough', #passthough features not listed
                    transformers=[('std', StandardTransformer , mixed_std_cols),('mm', MinMaxTransformer , mixed_minmax_cols)])
                X_Scaled = preprocessor.fit_transform(X)
                X_Scaled = pd.DataFrame(X_Scaled, columns = X.columns)
                self.activity_log_obj.log(file_object, 'Performed Min Max Scaling on selected {} columns and standard sclaing on {} columns'.format(','.join(mixed_minmax_cols),','.join(mixed_std_cols)))
            if response_column != None:
                return X_Scaled,y
            else:
                return X_Scaled
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception Occured. Exception message:  ' + str( e))
            self.activity_log_obj.log(file_object,'Scaling Unsuccessful.')
            raise Exception()
    
    
    def feature_engineering(self,file_object,data, unwanted_columns, Image_Folder, stage = 'Training'):
        try:
            df = data.copy()
            df = self.removing_unwanted_space(file_object,df)
            unwanted_columns = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year']
            df = self.removing_unwanted_columns(file_object,df,unwanted_columns)
            df.drop_duplicates(inplace = True)
            ### cleaning data ###
            ## replacing ? with NaN
            self.activity_log_obj.log(file_object, 'Cleaning data.')
            df.replace("?",np.NaN, inplace = True)
            self.activity_log_obj.log(file_object, 'Replaced all ? with NaN.')
            
            ## splitting necessary columns
            self.activity_log_obj.log(file_object, 'Splitting necessary columns.')
            df['policy_csl_used'] = df['policy_csl'].str.split('/').str[0].astype(int)
            df['policy_csl_used_available'] = df['policy_csl'].str.split('/').str[1].astype(int)
            df.drop(columns='policy_csl',inplace=True)
            self.activity_log_obj.log(file_object, 'Deleted the columns which were split. Policy CSL column was split')
            
            
            cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
            num_cols = [col for col in df.columns if df[col].dtypes != 'O']
            
            df_plot = df.copy()
            for col in cat_cols:
                plt.hist(df_plot[col].replace(np.NaN, 'NaN'))
                plt.title(col)
                plt.savefig(Image_Folder +'\\'+ col)
                plt.show()
                plt.clf()
            
            plt.figure(figsize=(13,8))
            corr_plot = sns.heatmap(df[num_cols].corr(), annot=True )
            #plt.savefig(corr_plot)
            plt.clf()
            
            #corr_plot.savefig('corr_plot.png')
            
            self.activity_log_obj.log(file_object, 'Label encoding categorical variables')
            df['insured_education_level'] = df['insured_education_level'].map({'JD' : 1, 'High School' : 2,'College':3,'Masters':4,'Associate':5,'MD':6,'PhD':7})
            df['incident_severity']       = df['incident_severity'].map({'Trivial Damage' : 1, 'Minor Damage' : 2,'Major Damage':3,'Total Loss':4})
            df['insured_sex']             = df['insured_sex'].map({'FEMALE' : 0, 'MALE' : 1})
            df['property_damage']         = df['property_damage'].map({'NO' : 0, 'YES' : 1})
            df['police_report_available'] = df['police_report_available'].map({'NO' : 0, 'YES' : 1})
            if stage  != 'Prediction':
                df['fraud_reported']      = df['fraud_reported'].map({'N' : 0, 'Y' : 1})
            df['collision_type']          = df['collision_type'].map({'Side Collision' : 0, 'Rear Collision' : 1, 'Front Collision' : 2})
            self.activity_log_obj.log(file_object, 'Label encoded categorical variables. The variables encoded are {}'.format(','.join(['insured_education_level','incident_severity','insured_sex','property_damage','police_report_available','collision_type'])))
            
            self.activity_log_obj.log(file_object, 'One hot encoding encoding categorical variables')
            df = pd.get_dummies(df, columns=['authorities_contacted'], prefix = ['ac'], drop_first=True)
            df = pd.get_dummies(df, columns=['incident_type'], prefix = ['it'], drop_first=True)
            df = pd.get_dummies(df, columns=['insured_relationship'], prefix = ['ir'], drop_first=True)
            df = pd.get_dummies(df, columns=['insured_occupation'], prefix = ['io'], drop_first=True)
            self.activity_log_obj.log(file_object, 'One hot encoded  categorical variables.The variables encoded are {}'.format(','.join(['authorities_contacted','incident_type','insured_relationship','insured_occupation'])))
    
            self.activity_log_obj.log(file_object, 'Imputing Null values')                                                                                                                 
            imputer = KNNImputer(n_neighbors=1)
            df_imputed = imputer.fit_transform(df)
            df_imputed = pd.DataFrame(df_imputed)
            df_columns = list(df.columns)
            df_imputed.columns = df_columns
            self.activity_log_obj.log(file_object, 'Imputed null values using KNN imputer') 
            
            df_imputed['collision_type'] = df_imputed['collision_type'].map({0.0:'Side Collision'  , 1.0:'Rear Collision', 2.0:'Front Collision'})
            df_imputed = pd.get_dummies(df_imputed, columns=['collision_type'], prefix = ['ct'], drop_first=True)
            
            self.activity_log_obj.log(file_object, 'Removing correlated columns')
            df_imputed = self.removing_unwanted_columns(file_object,df_imputed,['age','total_claim_amount'])
            if 'age' in num_cols:
                num_cols.remove('age')
            if 'total_claim_amount' in num_cols:
                num_cols.remove('total_claim_amount')
            self.activity_log_obj.log(file_object, 'Removed correlated columns. Columns removed are {}'.format(','.join(['age','total_claim_amount'])))
            return df_imputed,num_cols
        except  TypeError:
            self.activity_log_obj.log(file_object, "Error Occured while plotting :: %s" % OSError)
            raise TypeError
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception occured in remove_unwanted_spaces method of the Preprocessor class. Exception message:  ' + str( e))
            self.activity_log_obj.log(file_object,'unwanted space removal Unsuccessful. Exited the remove_unwanted_spaces method of the Preprocessor class')
            raise Exception()
            
        
if __name__ == "__main__":
    
    
    file_path = os.path.dirname(os.path.abspath("raw_data_validation"))
    data=pd.read_csv(file_path +'\\Data_Training\\input.csv')
    
    Image_Folder = file_path +'\\Images\\' 
    Path(Image_Folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(Image_Folder))

    
    file_object = open("Training_Logs/Activity_Log.txt", 'a+')
    unwanted_columns = ['policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date','incident_state','incident_city','insured_hobbies','auto_make','auto_model','auto_year']
    data_preprocessing_obj = data_preprocessing()
    
    final_data,num_cols = data_preprocessing_obj.feature_engineering(file_object,data, unwanted_columns,Image_Folder)
    mixed_minmax_cols = ['insured_education_level','incident_severity','property_damage'] 
    mixed_std_cols = num_cols
    X,y = data_preprocessing_obj.feature_scaling_splitting(file_object,final_data,'fraud_reported','Mixed',mixed_minmax_cols , mixed_std_cols)
    final_data.head()
    
    file_object.close()
            