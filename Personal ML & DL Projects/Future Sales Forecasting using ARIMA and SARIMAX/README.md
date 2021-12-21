# Future Sales Forecasting using ARIMA and SARIMAX

Data Details
  1)	The data in consideration is a sales data for perrin-freres-monthly-champagne.
  2)	There are 2 columns where one is a date column and other is the sales column.
  3)	The data ranges from 1964 to 1972.

Steps Implemented in Model
Step 1: Imported the data into a pandas dataframe and performed data cleaning on all features. The steps 
             include renaming columns, changing datatypes, and replacing nulls with blank space.
             
Step 2: Performed the test for Stationarity using Augmented Dickey Fueler test. 

Step 3: Performed differencing on the data to eliminate seasonality

Step 4: Performed auto-correlation to calculate p, d and q.

Step 4: Developed and compared ARIMA and SARIMAX model.

Step 5: Predicted future sales using the better model.



Files used to run the model
1)	Future Sales Forecasting using ARIMA and SARIMAX.ipynb

3)	perrin-freres-monthly-champagne-.csv
