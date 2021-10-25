import sqlite3
import pandas as pd
import numpy as np
import os
import shutil
import glob
import csv
from activity_logger import activity_logger
from raw_data_validation import raw_data_validation
from database_activity import database_activity
from pathlib import Path
import pickle


class folder_management_class():
    def __init__(self):
        self.activity_log_obj = activity_logger()
        
    def author(self):
        return 'Shrikanth Mahale'
        
    def training_folder_management(self,good_data_folder,bad_data_folder,good_archive,bad_archive,file_object):
        try:
            ##Dealing with bad files
            self.activity_log_obj.log(file_object, "Archiving Bad Files")
            bad_files = glob.glob(bad_data_folder+"*.csv", recursive=True)
            if len(bad_files) > 0:
                for bad_file in bad_files:
                    shutil.move(bad_file, bad_archive)
                    self.activity_log_obj.log(file_object, "Moved the file {} to the bad data folder".format(bad_file))
            else:
                self.activity_log_obj.log(file_object, "There are no bad files to Archive")
            
            ##Dealing with good files
            self.activity_log_obj.log(file_object, "Archiving Good Files")
            good_files = glob.glob(good_data_folder+"*.csv", recursive=True)
            if len(good_files) > 0:
                for good_file in good_files:
                    shutil.move(good_file, good_archive)
                    self.activity_log_obj.log(file_object, "Moved the file {} to the good data folder".format(good_file))
            else:
                self.activity_log_obj.log(file_object, "There are no good files to Archive")
        except  OSError:
            self.activity_log_obj.log(file_object, "Error Occured while moving the file :: %s" % OSError)
            raise OSError
        except  AttributeError:
            self.activity_log_obj.log(file_object, "Error Occured :: %s" % AttributeError)
            raise OSError
        except  Exception  as e:
            self.activity_log_obj.log(file_object, "Error Occured:: %s" % e)
            raise e

    def model_folder_management(self,model_folder,model_folder_name,model_name,model,file_object):
        try:
            model_name_folder = model_folder + model_folder_name +'\\'
            Path(model_name_folder).mkdir(parents=True, exist_ok=True)
            self.activity_log_obj.log(file_object, "Created {} folder to save the models best estimator".format(model_name))
            with open(model_name_folder + model_name+'.sav','wb') as f:
                pickle.dump(model, f) # save the model to file
            self.activity_log_obj.log(file_object, "Model saved in the path {}".format(model_name_folder))
        except  OSError:
            self.activity_log_obj.log(file_object, "Error Occured while saving the file :: %s" % OSError)
            raise OSError
        except  AttributeError:
            self.activity_log_obj.log(file_object, "Error Occured :: %s" % AttributeError)
            raise OSError
        except  Exception  as e:
            self.activity_log_obj.log(file_object, "Error Occured:: %s" % e)
            raise e
            
if __name__ == "__main__":
    
    file_object = open("Training_Logs/Activity_Log.txt", 'a+')
    
    json_file = 'schema_training.json'
    
    file_path = os.path.dirname(os.path.abspath("training_validation_insertion"))
    input_folder = file_path+ '\\Training_Batch_Files\\' 
    
    good_data_folder = file_path +'\\Good_Data\\' 
    Path(good_data_folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(good_data_folder))
    
    
    bad_data_folder = file_path +'\\Bad_Data\\' 
    Path(bad_data_folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(bad_data_folder))
    
    Database_Folder = file_path+ '\\Database\\'
    Path(Database_Folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(Database_Folder))
    
    archive_folder = file_path+ '\\Archive\\'
    Path(archive_folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(archive_folder))
    
    
    good_archive = archive_folder+ 'Good_Data\\'
    Path(good_archive).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(good_archive))
    
    bad_archive = archive_folder+ 'Bad_Data\\'
    Path(bad_archive).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(bad_archive))
    
    model_folder = file_path +'\\Model\\' 
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(model_folder))
    
    DatabaseName = 'Training'
    table_name  = 'tbl_training'

    file_object = open("Training_Logs/Activity_Log.txt", 'a+')
    raw_data_obj = raw_data_validation()
    database_activity_obj = database_activity()
    folder_management_class_obj = folder_management_class()
    folder_management_class_obj.training_folder_management(good_data_folder,bad_data_folder,good_archive,bad_archive)
    file_object.close()
    