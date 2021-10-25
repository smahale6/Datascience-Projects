import sqlite3
import pandas as pd
import numpy as np
import os
import shutil
import glob
import csv
from activity_logger import activity_logger
from raw_data_validation import raw_data_validation
from pathlib import Path

class database_activity:
    def __init__(self):
        self.activity_log_obj = activity_logger()
        
    def author(self):
        return 'Shrikanth Mahale'
        
    def database_creation(self,Path,DatabaseName,file_object): 
        try:
            conn = sqlite3.connect(Path+DatabaseName+'.db')
            self.activity_log_obj.log(file_object, "Opened %s database successfully" % DatabaseName)
        except ConnectionError:
            self.activity_log_obj.log(file_object, "Error while connecting to database: %s" %ConnectionError)
            raise ConnectionError
        return conn

    def data_etl(self,Database_Folder,good_data_folder,file_object,DatabaseName,column_names,column_datatypes,table_name,data_folder):
        try:
            ### DATA TRANSFORMATION ###
            files = glob.glob(good_data_folder+"fraud*.csv", recursive=True)
            df_combined = pd.DataFrame()
            for filepathname in files:
                self.activity_log_obj.log(file_object, "Imported {}".format(filepathname))
                df = pd.read_csv(filepathname)
                df_combined = pd.concat([df_combined,df])
            df_combined_str_cols = [cols  for cols in df_combined.columns if df_combined[cols].dtypes == 'O'] 
            for col in df_combined_str_cols:
                df_combined[col] = df_combined[col].apply(lambda x:"'" + str(x) +"'")
            df_combined.to_csv(Database_Folder + "\\transformed_file.csv" , index=None, header=True)
            self.activity_log_obj.log(file_object, "%s: Quotes added successfully!!")
            
            ### TABLE CREATION ###
            conn = self.database_creation(Database_Folder,DatabaseName,file_object)
            c=conn.cursor()
            self.activity_log_obj.log(file_object, "Connection to the database {} has been established".format(DatabaseName))
            col_name_dtype = ''
            for record in range(0,len(column_names)):
                col_name_dtype = col_name_dtype + column_names[record] + ' '  + column_datatypes[record] +' ,'
            col_name_dtype = col_name_dtype[:-1]
            create_query = 'CREATE TABLE IF NOT EXISTS '+ table_name +' (' + col_name_dtype +')'
            self.activity_log_obj.log(file_object, "Table {} has been successfully created".format(table_name))
            c.execute(create_query)
            
            
            ### DATA INSERTION ###
            table_data = pd.read_sql_query("SELECT distinct policy_number from " + table_name, conn)
            insert_data = df_combined.copy()
            policy_number = list(table_data['policy_number'])
            
            onlyfiles = glob.glob(Database_Folder + "trans*.csv", recursive=True)
            for file in onlyfiles:
                with open(file, "r") as f:
                    next(f)
                    reader = csv.reader(f, delimiter="\n")
                    for line in enumerate(reader):
                        for list_ in (line[1]):
                            try:
                                conn.execute('INSERT INTO ' + table_name + ' values ({values})'.format(values=(list_)))
                                self.activity_log_obj.log(file_object, "Data has been inserted successfully. Total {} records are inserted in {} table".format(len(df_combined),table_name))
                                conn.commit()
                            except Exception as e:
                                self.activity_log_obj.log(file_object, "Error Occured:: %s" % e)
                                raise e
                                
            ### EXPORT TO CSV ###
            input_file = pd.read_sql_query("SELECT * FROM " + table_name , conn)
            input_file.to_csv(data_folder+'input.csv', index = False)
            self.activity_log_obj.log(file_object, "Exported Input.csv from the database {}".format(DatabaseName))
            c.close()
            conn.close()
            
            self.activity_log_obj.log(file_object, "Connection to the database {} has been closed".format(DatabaseName))
        except Exception as e:
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
    
    data_folder = file_path+ '\\Data_Training\\' 
    Path(data_folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(data_folder))
    
    bad_data_folder = file_path +'\\Bad_Data\\' 
    Path(bad_data_folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(bad_data_folder))
    
    Database_Folder = file_path+ '\\Database\\'
    Path(Database_Folder).mkdir(parents=True, exist_ok=True)
    print('Created Folder {}'.format(Database_Folder))
    
    
    DatabaseName = 'Training'
    table_name  = 'tbl_training'

    file_object = open("Training_Logs/Activity_Log.txt", 'a+')
    raw_data_obj = raw_data_validation()
    database_activity_obj = database_activity()
    LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns, column_names, column_datatypes = raw_data_obj.read_json_file(json_file)
    raw_data_obj.input_file_validation(input_folder,json_file, good_data_folder, bad_data_folder, file_object)
    database_activity_obj.data_etl(Database_Folder,good_data_folder,file_object,DatabaseName,column_names,column_datatypes,table_name,data_folder)
    file_object.close()
    
   
    
   
    
   
    
   
    
   
    
   
    
   
'''
columns = ''
for cols in range(0, len(column_names)):
    columns = columns + '?,'
columns = columns[:-1]

conn = database_creation(Path,DatabaseName)
table_creation(DatabaseName,column_names, column_datatypes, table_name)
c=conn.cursor()


with open(Database_Folder+'transformed_file.csv') as f:
    reader = csv.reader(f)
    for field in reader:
        c.execute("INSERT INTO " + table_name + " VALUES (" + columns + ");" , field)





query=" delete from  tbl_training"
conn.execute(query)
conn.commit()

conn = sqlite3.connect('Training.db')
c = conn.cursor()

df = pd.read_sql_query("SELECT * FROM tbl_training", conn)
transformed_file = pd.read_csv(Database_Folder +'transformed_file.csv', index_col=None )
transformed_file.columns = df.columns
'''