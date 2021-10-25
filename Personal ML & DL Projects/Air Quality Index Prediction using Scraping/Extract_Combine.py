from Plot_AQI import import_pm
from bs4 import BeautifulSoup
import requests
import sys
import pandas as pd
import time
import numpy as np



def Meta_Data(Month, Year):
    HTML_Path = '/Data/HTML_Data' 
    def multiples(m, max_value,count = 1000):
        total_multiples = []
        for i in range(1,count):
            multiple = i*m
            if multiple > max_value:
                break
            else:
                total_multiples.append(multiples)
        return len(total_multiples)
    
    
    file_html = open(HTML_Path+'/'+str(Year)+'/'+str(Month)+'.html','rb')
    plain_text = file_html.read()
    
    tempD = []
    finalD = []
    
    soup = BeautifulSoup(plain_text,'lxml')
    for table in soup.findAll('table',{'class':'medias mensuales numspan'}):
        for tbody in table:
            for tr in tbody:
                a = tr.get_text()
                tempD.append(a)
                    
        
    Column_Names = tempD[:15]
    Row_Values = tempD[15:]
    Value_List = []
    
    counts = multiples(len(Column_Names), len(Row_Values),count = 1000)
    for count in range(counts):
        row_value = []
        if len(Row_Values)>0:
            for index in range(0,len(Column_Names)):
                row_value.append(Row_Values[index])
            Value_List.append(row_value)
            if len(Row_Values) > 15:
                del Row_Values[:15]
            else:
                Row_Values.clear()
        else:
            break
        
    df = pd.DataFrame(Value_List,columns=Column_Names)
    df['Month'] = str(Month)
    df['Year'] = str(Year)
    df = df[df.Day != 'Monthly means and totals:']
    return df

def combined_data(start_year,end_year, path):
    combined_data = pd.DataFrame()
    for year in range(start_year,end_year+1):
        for month in range(1,13):
            df = Meta_Data(month, year)
            combined_data = combined_data.append(df) 
    combined_data['Date'] = combined_data['Year']+'-'+combined_data['Month']+'-'+combined_data['Day']
    combined_data['Date'] =  pd.to_datetime(combined_data['Date'])
    combined_data.reset_index(inplace=True,drop=True)
    combined_data = combined_data.loc[:,['Date','T','TM','Tm','H','PP','VV','V','VM']]    
    combined_data_obj = combined_data.select_dtypes(['object'])
    combined_data[combined_data_obj.columns] = combined_data_obj.apply(lambda x: x.str.strip())
    combined_data.replace('', np.nan, inplace= True)
    # combined_data = combined_data.dropna(how='all')
    combined_data.replace('-', np.nan, inplace= True)
    
    combined_data = combined_data.loc[combined_data['T'].notna(),:]
    return combined_data
        
if __name__ == "__main__":
    start_time=time.time()
    HTML_Path = '/Data/HTML_Data'  
    Aqi_Data = import_pm()
    Variable_Data = combined_data(2013,2018, HTML_Path)
    All_Data = pd.merge(Variable_Data,Aqi_Data,how = 'inner',on = 'Date')
    Final_Data = All_Data.loc[:,['T','TM','Tm','H','PP','VV','V','VM','PM2.5']]
    #Final_Data = Final_Data.loc[Final_Data['PM2.5'].notna()]
    stop_time=time.time()
    print("Time taken {}".format(stop_time-start_time))        
        
    

