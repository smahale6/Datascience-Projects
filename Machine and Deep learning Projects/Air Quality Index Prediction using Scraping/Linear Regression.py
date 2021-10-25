import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path

from Html_script import retrieve_html
from Plot_AQI import import_pm
from Extract_Combine import combined_data,Meta_Data



start_time=time.time()

HTML_Path = 'C:/Users/D100793/OneDrive - Citizens/Desktop/Krish Naik/Live Implementation Practice/Data/HTML_Data'  
Path(HTML_Path).mkdir(parents=True, exist_ok=True)
print('Created Folder {}'.format(HTML_Path))

retrieve_html()
Aqi_Data = import_pm()
Variable_Data = combined_data(2013,2018, HTML_Path)
All_Data = pd.merge(Variable_Data,Aqi_Data,how = 'inner',on = 'Date')
Final_Data = All_Data.loc[:,['T','TM','Tm','H','PP','VV','V','VM','PM2.5']]