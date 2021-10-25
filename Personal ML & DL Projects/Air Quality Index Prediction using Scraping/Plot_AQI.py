import pandas as pd
import matplotlib.pyplot as plt
import pandasql as ps
import re
import numpy as np
import datetime
import glob
import time

AQI_path = r'C:\\Users\\D100793\\OneDrive - Citizens\\Desktop\\Krish Naik\\Live Implementation Practice\\Air Quality Index Prediction\\Data\\AQI'
#AQI_path = '/Data/AQI'

def import_pm():
    Aqi = pd.DataFrame()
    files = glob.glob(AQI_path+"\\aqi*.csv", recursive=True)
    for filename in files:
        df = pd.read_csv(filename)
        Aqi = Aqi.append(df)
    Aqi.reset_index(inplace=True,drop=True)
    Aqi['PM2.5'] = Aqi['PM2.5'].replace('NoData', np.nan)
    Aqi['PM2.5'] = Aqi['PM2.5'].replace('PwrFail', np.nan)
    Aqi['PM2.5'] = Aqi['PM2.5'].replace('---', np.nan)
    Aqi['PM2.5'] = Aqi['PM2.5'].replace('InVld', np.nan)
    Aqi['PM2.5 AQI'] = Aqi['PM2.5 AQI'].replace('NoData', np.nan)
    Aqi['PM2.5 AQI'] = Aqi['PM2.5 AQI'].replace('PwrFail', np.nan)
    Aqi['PM2.5 AQI'] = Aqi['PM2.5 AQI'].replace('---', np.nan)
    Aqi['PM2.5 AQI'] = Aqi['PM2.5 AQI'].replace('InVld', np.nan)
    Aqi['PM2.5']     = Aqi['PM2.5'].astype(float)
    Aqi['PM2.5 AQI'] = Aqi['PM2.5 AQI'].astype(float)
    Aqi['Date'] = Aqi['Date'].str.extract(r"(\d{1,2}[/. ](?:\d{1,2}|January|Jan)[/. ]\d{2}(?:\d{2})?)")
    Aqi['Date'] =  pd.to_datetime(Aqi['Date'])
    Aqi.to_csv("AQIALL.csv")
    
    Aqi_Daily = Aqi.groupby('Date')["PM2.5","PM2.5 AQI"].mean()
    Aqi_Daily.reset_index(inplace=True)
    Aqi_Daily['Month'] = Aqi_Daily.Date.dt.month
    Aqi_Daily['Day'] = Aqi_Daily.Date.dt.day
    Aqi_Daily['Year'] = Aqi_Daily.Date.dt.year
    Aqi_Daily.to_csv("Aqi_Daily.csv")
    return Aqi_Daily

if __name__ == "__main__":
    start_time=time.time()
    Aqi_Daily = import_pm()
    stop_time=time.time()
    print("Time taken {}".format(stop_time-start_time))
            
            
                