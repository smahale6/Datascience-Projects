from datetime import datetime
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
import os
import shutil
from raw_data_validation import raw_data_validation
from database_activity import database_activity
from activity_logger import activity_logger
from folder_management import folder_management_class

class prediction_validation_insertion_class:
    def __init__(self):
        self.activity_log_obj = activity_logger()
        self.file_object = open("Prediction_Logs/Activity_Log.txt", 'a+')
        self.database_activity_obj = database_activity()
        self.raw_data_validation_obj = raw_data_validation()
        self.json_file = 'schema_prediction.json'
        self.file_path = os.path.dirname(os.path.abspath("prediction_validation_insertion"))
        self.input_folder = self.file_path+ '\\Prediction_Batch_Files\\'
        self.folder_management_class_obj = folder_management_class()
        
    def author(self):
        return 'Shrikanth Mahale'
        
    def prediction_validation_insertion(self,database_name,table_name):
        try:
            self.activity_log_obj.log(self.file_object, 'Start of Validation on files for prediction!!')
            
            good_data_folder = self.file_path +'\\Good_Data\\' 
            Path(good_data_folder).mkdir(parents=True, exist_ok=True)
            self.activity_log_obj.log(self.file_object, 'Created Good Data Folder')
            
            bad_data_folder = self.file_path +'\\Bad_Data\\' 
            Path(bad_data_folder).mkdir(parents=True, exist_ok=True)
            self.activity_log_obj.log(self.file_object, 'Created Bad Data Folder')
            
            Database_Folder = self.file_path+ '\\Database\\'
            Path(Database_Folder).mkdir(parents=True, exist_ok=True)
            self.activity_log_obj.log(self.file_object, 'Accessed/Created Database Folder')
            
            Archive_Folder = self.file_path+ '\\Archive\\'
            Path(Archive_Folder).mkdir(parents=True, exist_ok=True)
            self.activity_log_obj.log(self.file_object, 'Accessed Archive Folder')
            
            good_archive = Archive_Folder + '\\Good_Data\\'
            Path(good_archive).mkdir(parents=True, exist_ok=True)
            self.activity_log_obj.log(self.file_object, 'Accessed Good Data Folder in Archive Folder')
            
            bad_archive = Archive_Folder + '\\Bad_Data\\'
            Path(bad_archive).mkdir(parents=True, exist_ok=True)
            self.activity_log_obj.log(self.file_object, 'Accessed Bad Data Folder in Archive Folder')
            
            data_folder = self.file_path+ '\\Data_Prediction\\' 
            Path(data_folder).mkdir(parents=True, exist_ok=True)
            self.activity_log_obj.log(self.file_object, 'Accessed Data Folder')
            
            self.activity_log_obj.log(self.file_object, 'Starting Raw Data Validation')
            LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, column_names, column_datatypes = self.raw_data_validation_obj.read_json_file(self.json_file)
            self.raw_data_validation_obj.input_file_validation(self.input_folder,self.json_file, good_data_folder, bad_data_folder, self.file_object)
            self.activity_log_obj.log(self.file_object, 'Completed Raw Data Validation')
            
            self.activity_log_obj.log(self.file_object, 'Starting Database Operations')
            self.database_activity_obj.data_etl(Database_Folder,good_data_folder,self.file_object,database_name,column_names,column_datatypes,table_name,data_folder)
            self.activity_log_obj.log(self.file_object, 'Completed Raw Database Operations')
            
            self.activity_log_obj.log(self.file_object, 'Starting Folder Management Operations for prediction data')
            self.folder_management_class_obj.training_folder_management(good_data_folder,bad_data_folder,good_archive,bad_archive,self.file_object)
            self.activity_log_obj.log(self.file_object, 'Starting Folder Management Operations for prediction data')
            
            self.activity_log_obj.log(self.file_object, 'Validation on files for prediction Completed!!')
            
            self.file_object.close()
            
        except Exception as e:
            raise e
            
if __name__ == "__main__":
    prediction_validation_insertion_obj = prediction_validation_insertion_class()
    prediction_validation_insertion_obj.prediction_validation_insertion('Insurance','tbl_for_prediction')