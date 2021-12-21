import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
import glob
import os
import shutil
from activity_logger import activity_logger

class raw_data_validation:
    def __init__(self):
        self.activity_log_obj = activity_logger()
        
    def author(self):
        return 'Shrikanth Mahale'

    def read_json_file(self,json_file):
        f = open(json_file)
        data = json.load(f)
        LengthOfDateStampInFile = data['LengthOfDateStampInFile']
        LengthOfTimeStampInFile = data['LengthOfTimeStampInFile']
        NumberofColumns = data['NumberofColumns']
        column_names = list(data['ColName'].keys())
        column_datatypes = list(data['ColName'].values())
        return LengthOfDateStampInFile,LengthOfTimeStampInFile,NumberofColumns,column_names,column_datatypes

    
    def input_file_validation(self,input_folder,json_file, good_data_folder, bad_data_folder, file_object):
        try:
            files = glob.glob(input_folder+"*.csv", recursive=True)
            LengthOfDateStampInFile,LengthOfTimeStampInFile,NumberofColumns,column_names,column_datatypes = self.read_json_file(json_file)
            for filepath in files:
                filename = filepath.split('\\')[-1].split('.')[0]
                fileformat = filepath.split('\\')[-1].split('.')[1]
                self.activity_log_obj.log(file_object,'Looking at the file ' + filename + '.' + fileformat)
                if filename.split('_')[0] == 'visibility' and len(filename.split('_')[1]) == LengthOfDateStampInFile and len(filename.split('_')[2]) == LengthOfTimeStampInFile and fileformat == 'csv':
                    self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' is named in correct format')
                    df = pd.read_csv(filepath)
                    df_cols = list(df.columns)
                    if df_cols == column_names and len(df_cols) == NumberofColumns:
                        self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' has correct columnname and number of columns.')
                        if len(df.columns[df.isnull().any()].tolist()) == 0:
                            shutil.move(filepath, good_data_folder)
                            self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' has no all null columns and number of columns.')
                            self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' has been moved to Good Data Folder')
                        else:
                            self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' has few all null columns and number of columns.')
                            shutil.move(filepath, bad_data_folder) 
                            self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' has been moved to Bad Data Folder')
                    else:
                        self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' has incorrect columnes or incorrect number of columns.')
                        shutil.move(filepath, bad_data_folder)
                        self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' has been moved to Bad Data Folder')
                else:
                    self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' is named in incorrect format.')
                    shutil.move(filepath, bad_data_folder)
                    self.activity_log_obj.log(file_object,'The file ' + filename + '.' + fileformat + ' has been moved to Bad Data Folder')
        except  OSError:
            self.activity_log_obj.log(file_object, "Error Occured while moving the file :: %s" % OSError)
            raise OSError
        except  Exception  as e:
            self.activity_log_obj.log(file_object, "Error Occured:: %s" % e)
            raise e
            


if __name__ == "__main__":
    json_file = 'schema_training.json'
    
    file_path = os.path.dirname(os.path.abspath("raw_data_validation"))
    input_folder = file_path+ '\\Training_Batch_Files\\' 
    
    good_data_folder = file_path +'\\Good_Data\\' 
    Path(good_data_folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(good_data_folder))
    
    
    bad_data_folder = file_path +'\\Bad_Data\\' 
    Path(bad_data_folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(bad_data_folder))
    
    file_object = open("Training_Logs/Activity_Log.txt", 'a+')
    raw_data_obj = raw_data_validation()
    LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, column_names, column_datatypes = raw_data_obj.read_json_file(json_file)
    raw_data_obj.input_file_validation(input_folder,json_file, good_data_folder, bad_data_folder, file_object)
    file_object.close()
            



    



