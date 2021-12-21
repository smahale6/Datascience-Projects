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
            if scaling_type not in ['MinMax','Standard','Mixed','NoScale']:
                raise ValueError("Only allowed Values are 'MinMax','Standard','Mixed' and 'NoScale'")
            elif scaling_type == 'MinMax':
                X_Scaled = MinMaxTransformer.fit_transform(X)
                X_Scaled = pd.DataFrame(data=X_Scaled, columns=X.columns,index=X.index)
                self.activity_log_obj.log(file_object, 'Performed MinMax Scaling on all columns')
            elif scaling_type == 'Standard':
                X_Scaled = StandardScaler.fit_transform(X)
                X_Scaled = pd.DataFrame(data=X_Scaled, columns=X.columns,index=X.index)
                self.activity_log_obj.log(file_object, 'Performed Standard Scaling on all columns')
            elif scaling_type == 'Mixed':
                preprocessor = ColumnTransformer(
                    remainder='passthrough', #passthough features not listed
                    transformers=[('std', StandardTransformer , mixed_std_cols),('mm', MinMaxTransformer , mixed_minmax_cols)])
                X_Scaled = preprocessor.fit_transform(X)
                X_Scaled = pd.DataFrame(X_Scaled, columns = X.columns)
                self.activity_log_obj.log(file_object, 'Performed Min Max Scaling on selected {} columns and standard sclaing on {} columns'.format(','.join(mixed_minmax_cols),','.join(mixed_std_cols)))
            else:
                X_Scaled = X
            if response_column != None:
                return X_Scaled,y
            else:
                return X_Scaled
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception Occured. Exception message:  ' + str( e))
            self.activity_log_obj.log(file_object,'Scaling Unsuccessful.')
            raise Exception()
    
    
    def feature_engineering(self,file_object,data, Image_Folder, stage = 'Training'):
        try:
            df = data.copy()
            unwanted_columns = ['DATE','Precip']
            df = self.removing_unwanted_columns(file_object,df,unwanted_columns)
            df.drop_duplicates(inplace = True)
            ### cleaning data ###
            cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
            num_cols = [col for col in df.columns if df[col].dtypes != 'O']
            
            self.activity_log_obj.log(file_object, 'Saving Distribution Plots')
            df_plot = df.copy()
            plt.figure(figsize=(20,25), facecolor='white')
            plotnumber = 1
            for column in df_plot:
                ax = plt.subplot(4,3,plotnumber)
                sns.distplot(df_plot[column])
                plt.xlabel(column,fontsize=20)
                plotnumber += 1
                plt.tight_layout()
            plt.savefig(Image_Folder +'\\'+ 'Distributions_'+ stage)
            plt.clf()
            self.activity_log_obj.log(file_object, 'Saved Distribution Plots')
            
            self.activity_log_obj.log(file_object, 'Saving Correlation Plots')
            if stage == 'Training':
                df_plot = df_plot.drop(['VISIBILITY'],axis = 1)
            sns.heatmap(df_plot.corr().abs(), vmin = -0.5, vmax = 1,annot=True)
            plt.savefig(Image_Folder +'\\'+ 'Corr_Plot_' + stage)
            plt.clf()
            self.activity_log_obj.log(file_object, 'Saved Correlation Plots')
            self.activity_log_obj.log(file_object, 'Removing correlated columns')
            df = self.removing_unwanted_columns(file_object,df,['WETBULBTEMPF','DewPointTempF','StationPressure'])
            self.activity_log_obj.log(file_object, 'Removed correlated columns. Columns removed are {}'.format(','.join(['WETBULBTEMPF','DewPointTempF','StationPressure'])))
            return df
        except  TypeError:
            self.activity_log_obj.log(file_object, "Error Occured while plotting :: %s" % OSError)
            raise TypeError
        except Exception as e:
            self.activity_log_obj.log(file_object,'Exception occured in remove_unwanted_spaces method of the Preprocessor class. Exception message:  ' + str( e))
            self.activity_log_obj.log(file_object,'unwanted space removal Unsuccessful. Exited the remove_unwanted_spaces method of the Preprocessor class')
            raise Exception()
            
        
if __name__ == "__main__":
    
    
    file_path = os.path.dirname(os.path.abspath("raw_data_validation"))
    data = pd.read_csv(file_path +'\\Data_Prediction\\input.csv')
    
    Image_Folder = file_path +'\\Images\\' 
    Path(Image_Folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(Image_Folder))

    
    file_object = open("Prediction_Logs/Activity_Log.txt", 'a+')
    data_preprocessing_obj = data_preprocessing()
    
    final_data = data_preprocessing_obj.feature_engineering(file_object,data, Image_Folder) 
    X,y = data_preprocessing_obj.feature_scaling_splitting(file_object,final_data,'VISIBILITY','NoScale')
    final_data.head()
    file_object.close()
            

    