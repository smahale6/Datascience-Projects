import pandas as pd
import os
import pickle
from pathlib import Path
from activity_logger import activity_logger
from preprocessing import data_preprocessing
from KMeansClustering import KMeansClustering
from Model_Tuner import Model_Tuner
from folder_management import folder_management_class
from prediction_validation_insertion import prediction_validation_insertion_class
from sklearn.model_selection import train_test_split
import glob
from datetime import datetime


class Predict_From_Model:
    def __init__(self):
        self.activity_log_obj = activity_logger()
        self.file_object = open("Prediction_Logs/Activity_Log.txt", 'a+')
        self.data_preprocessing_obj = data_preprocessing()
        self.Folder_Management_obj = folder_management_class()
        self.json_file = 'schema_prediction.json'
        self.file_path = os.path.dirname(os.path.abspath("Predict_From_Model"))
        self.Model_Folder = self.file_path +'\\Model\\' 
        self.Image_Folder_Prediction = self.file_path +'\\Images\\' 
        self.Prediction_Output = self.file_path +'\\Prediction_Output\\' 
    
    def author(self):
        return 'Shrikanth Mahale'
        
    def load_model(self,foldername,filename):
        self.activity_log_obj.log(self.file_object,  'Entered the load_model method of the File_Operation class')
        try:
            with open(self.Model_Folder  + foldername + '\\' + filename + '.sav','rb') as f:
                self.activity_log_obj.log(self.file_object,'Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
                return pickle.load(f)
        except Exception as e:
            self.activity_log_obj.log(self.file_object,'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.activity_log_obj.log(self.file_object,'Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise Exception()
    
    def get_correct_model(self,folder_name):
        try:
            self.activity_log_obj.log(self.file_object,  'Finding best estimator')
            files = glob.glob(self.Model_Folder + folder_name+ "\\" +"*.sav", recursive=True)
            best_estimator = files[0].split('\\')[-1].split('.')[0]
            self.activity_log_obj.log(self.file_object,  'The best estimator is {}'.format(best_estimator))
            return best_estimator
        except Exception as e:
            self.activity_log_obj.log(self.file_object,'Exception occured in get_correct_model method of the Model_Finder class. Exception message:  ' + str(e))
            raise Exception()
    
    def prediction(self):
        self.activity_log_obj.log(self.file_object, 'Start of Prediction of data!!')
        try:
            self.activity_log_obj.log(self.file_object, 'Importing data!!')
            data = pd.read_csv(self.file_path +'\\Data_Prediction\\input.csv')
            ### Preprocessing Data  ###
            self.activity_log_obj.log(self.file_object, 'Start of Preprocessing of data!!')
            Path(self.Image_Folder_Prediction).mkdir(parents=True, exist_ok=True)
            final_data = self.data_preprocessing_obj.feature_engineering(self.file_object,data,self.Image_Folder_Prediction,'Prediction')
            self.activity_log_obj.log(self.file_object, 'End of Preprocessing of data!!')
            ### Scaling Data  ###
            self.activity_log_obj.log(self.file_object, 'Start of Scaling of data!!')
            X = self.data_preprocessing_obj.feature_scaling_splitting(self.file_object,final_data,None,'NoScale')
            ### Clustering of Data  ###
            self.activity_log_obj.log(self.file_object, 'Start of Clustering of data!!')
            kmeans = self.load_model('Kmeans','Kmeans')
            clusters = kmeans.predict(X)
            X['Cluster'] = clusters
            Tot_Clusters=len(list(X['Cluster'].unique()))
            self.activity_log_obj.log(self.file_object, 'End of Clustering of data!!')
            predictions = []
            for i in range(0,Tot_Clusters):
                self.activity_log_obj.log(self.file_object, 'Running Model for cluster number {}'.format(i))
                cluster_features = X[X['Cluster']==i] # filter the data for one cluster
                cluster_features = cluster_features.drop(['Cluster'],axis=1)
                cluster_folder_name = 'Cluster ' + str(i)
                best_estimator = self.get_correct_model(cluster_folder_name)
                reg = self.load_model(cluster_folder_name,best_estimator)  
                result  =(reg.predict(cluster_features))
                for res in result:
                    predictions.append(res)
            Path(self.Prediction_Output).mkdir(parents=True, exist_ok=True)
            final = pd.DataFrame(predictions, columns=['Predictions'])
            now = str(datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
            final.to_csv(self.Prediction_Output + "predictions" + ".csv",header=True) #appends result to prediction file
            self.activity_log_obj.log(self.file_object, 'Prediction Completed!!')      
            self.file_object.close()
        except Exception as e:
            raise e
            
            
if __name__ == "__main__":
    prediction_validation_insertion_obj = prediction_validation_insertion_class()
    prediction_obj = Predict_From_Model()
    
    prediction_validation_insertion_obj.prediction_validation_insertion('WeatherVisibility','tbl_prediction')
    prediction_obj.prediction()
