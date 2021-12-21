import pandas as pd
import os
from pathlib import Path
from activity_logger import activity_logger
from preprocessing import data_preprocessing
from KMeansClustering import KMeansClustering
from Model_Tuner import Model_Tuner
from folder_management import folder_management_class
from training_validation_insertion import train_validation_insertion_class
from sklearn.model_selection import train_test_split


class Train_Model:
    def __init__(self):
        self.activity_log_obj = activity_logger()
        self.file_object = open("Training_Logs/Activity_Log.txt", 'a+')
        self.data_preprocessing_obj = data_preprocessing()
        self.KMeansClustering_obj = KMeansClustering()
        self.Model_Tuner_obj = Model_Tuner()
        self.Folder_Management_obj = folder_management_class()
        self.json_file = 'schema_training.json'
        self.file_path = os.path.dirname(os.path.abspath("Train_Model"))
        self.Image_Folder = self.file_path +'\\Images\\' 
        self.Model_Folder = self.file_path +'\\Model\\' 
        
    def author(self):
        return 'Shrikanth Mahale'
        
    def training_model(self):
        self.activity_log_obj.log(self.file_object, 'Start of Training of data!!')
        try:
            ### Importing Data  ###
            self.activity_log_obj.log(self.file_object, 'Importing data!!')
            data = pd.read_csv(self.file_path +'\\Data_Training\\input.csv')
            
            ### Preprocessing Data  ###
            self.activity_log_obj.log(self.file_object, 'Start of Preprocessing of data!!')
            Path(self.Image_Folder).mkdir(parents=True, exist_ok=True)
            final_data = self.data_preprocessing_obj.feature_engineering(self.file_object,data,self.Image_Folder)
            self.activity_log_obj.log(self.file_object, 'End of Preprocessing of data!!')
            
            ### Scaling Data  ###
            self.activity_log_obj.log(self.file_object, 'Start of Scaling of data!!')
            X,y = self.data_preprocessing_obj.feature_scaling_splitting(self.file_object,final_data,'VISIBILITY','NoScale')
            self.activity_log_obj.log(self.file_object, 'End of Scaling of data!!')
            
            ### Clustering of Data  ###
            self.activity_log_obj.log(self.file_object, 'Start of Clustering of data!!')
            Tot_Clusters = self.KMeansClustering_obj.elbow_diagram(self.file_object,X,self.Image_Folder)
            X,y,Kmeans = self.KMeansClustering_obj.cluster_creation(self.file_object,X,y,self.Image_Folder)
            self.Folder_Management_obj.model_folder_management(self.Model_Folder,'Kmeans','Kmeans',Kmeans,self.file_object)
            self.activity_log_obj.log(self.file_object, 'End of Clustering of data!!')
            
            
            X['Response'] = y
            
            for i in range(0,Tot_Clusters):
                self.activity_log_obj.log(self.file_object, 'Running Model for cluster number {}'.format(i))
                cluster_features = X[X['Cluster']==i] # filter the data for one cluster
                cluster_label = cluster_features['Response']
                cluster_features.drop(columns = 'Cluster', inplace = True)
                cluster_features.drop(columns = 'Response', inplace = True)
                ### Train Test Split  ###
                X_train, X_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=0.20, random_state=50)
                ### Getting best model ###
                self.activity_log_obj.log(self.file_object, 'Start of finding best model!!')
                hyperparameter_tuning_results = self.Model_Tuner_obj.model_hyperparameter_tuning(X_train,y_train,self.file_object)
                best_model,best_estimator =  self.Model_Tuner_obj.get_best_estimator(X_train,y_train,X_test,y_test,self.file_object,hyperparameter_tuning_results)
                self.activity_log_obj.log(self.file_object, 'The best model is {}!!'.format(best_model))
                ### Saving the best model ###
                cluster_folder_name = 'cluster ' + str(i)
                self.Folder_Management_obj.model_folder_management(self.Model_Folder,cluster_folder_name,best_model,best_estimator,self.file_object)
                
            self.activity_log_obj.log(self.file_object, 'Model Training Completed!!')
            
            self.file_object.close()
            
        except Exception as e:
            raise e
            
            
            
if __name__ == "__main__":
    train_validation_insertion_obj = train_validation_insertion_class()
    training_model_obj = Train_Model()
    
    train_validation_insertion_obj.train_validation_insertion('WeatherVisibility','tbl_training')
    training_model_obj.training_model()
            
            
            
            
            
            
            