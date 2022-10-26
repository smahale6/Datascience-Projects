# Insurance Fraud

> Given below are the code files that function for this project

1) activity_logger.py - Used to log every activity when the code is run. The class in this file is used in every other file to log activity.

2) raw_data_validation.py - Used to validate the format and the content of data files which includes name of the file and structure of the file.

3) database_activity.py - Used to manage activities in the SQLite3 database such as creating databases and tables. Also performs etl activity on data.

4) folder_management.py - Used to perform activities like creating folders and migrating files between folders.

5) training_validation_insertion.py - Central location to run raw_data_validation.py, database_activity.py and folder_management.py  using data in Training_Batch_Files folder for data validationa and insertion.

6) preprocessing.py - Used to perform exploratory data analysis and feature engineering of data. This includes removing NaN, encoding categorical variables, scaling numerical data etc.

7) KMeansClustering.py - Meant to locate cluster knee and create cluster on the input data provided.

8) Model_Tuner.py - Code for hyperparameter tuning using different classification models.

9) Train_Model.py - Central location to run the combination of preprocessing.py, KMeansClustering.py and Model_Tuner.py to preprocess and cluster the data, run the best model obtained through hyperparameterytuning on each cluster.

10) prediction_validation_insertion.py - Used to validate prediction data. Central location for running  raw_data_validation.py, database_activity.py and folder_management.py  using data in Prediction_Batch_Files folder for data validationa and insertion.

11) predict_from_model.py - Here every record in the data is assigned a cluster number from the clustering algorithm used in KMeansClustering step and the best model that was identified in Model_Tuner step is used to predict the y label.


> Given below are the JSON files used for this project
1) schema_training.json - Used as a reference to validate the file type, file name , column names and column data types of the data file that is used for training the model.

2) schema_prediction.json - Used as a reference to validate the file type, file name , column names and column data types of the data file that is used for predicting  the model.

> Given below are the txt files used for this project
1) requirements.txt - Includes all the python modules required for this project

> Given below are the folders setup for this project

1) Training_Batch_File - Used to store input data used for training which will be picked up by raw_data_validation.py for validating the data and file.

2) Prediction_Batch_File - Used to store input data used for training which will be picked up by raw_data_validation.py for validating the data and file.

3) Good_Data - Used to hold the validated data when the validation criteria has passed.

4) Bad Data - Used to hold the validated data when the validation criteria has failed.

5) Database - Used to store the database generated in the database_activity.py step.

6) Model - Used to hold models for each cluster

7) Images - Used to save the images gerenrated in preprocessing step.

8) Archives - Used to hold archives of old data.

9) Data_Training - Used to hold the csv used for training file exported from the database stored in the Database folder.

8) Data_Prediction - Used to hold the csv file used for production exported from the database stored in the Database folder.

9) Prediction_Output - Used to store the out of prediction after running the model of the data.

10) Training_Logs - Used to store the file that logs all the activites of a the code while training and validation the data.

11) Prediction_Logs - Used to store the file that logs all the activites of a the code while prediction and validation the data.



> Steps to run the model
1) Open Anaconda prompt and create a new environment using the command below. (Use Python 3.6)
	
	`conda create -n myenv python=3.6`

2) Activate the environment using the command below.

	`conda activate myenv`

3) Install all the packages mentioned in the requirement.txt file in the environment created using the command below.

	`pip install -r requirements.txt`

4) Save the raw input file in the Training_Batch_File folder.

5) Save the file to be predicted in the Prediction_Batch_File folder.

6) Run the file Train_Model.py

7) Run the file predict_from_model.py

8) Look at the Prediction_Output folder for output.
