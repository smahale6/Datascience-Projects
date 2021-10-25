import bs4 
from bs4 import BeautifulSoup
import os
import time
import pandas as pd
import numpy as np
import requests
import sys
from pathlib import Path


HTML_Path = '/Data/HTML_Data'  
Path(HTML_Path).mkdir(parents=True, exist_ok=True)
print('Created Folder {}'.format(HTML_Path))

def retrieve_html():
    for year in range(2013,2019):
        year_path = HTML_Path + '/' + str(year)
        Path(year_path).mkdir(parents=True, exist_ok=True)
        print('Created Folder {}'.format(year_path))
        for month in range(1,13):
            if month < 10:
                url = 'https://en.tutiempo.net/climate/0{}-{}/ws-432950.html'.format(month,year)
            else:
                url = 'https://en.tutiempo.net/climate/{}-{}/ws-432950.html'.format(month,year)
        
            texts = requests.get(url)
            texts_utf = texts.text.encode("UTF=8")
            
            
            with open(year_path+'/'+str(month)+'.html',"wb") as output:
                output.write(texts_utf)
                print("Downloaded Month {} for year {}".format(month,year))
        sys.stdout.flush()


if __name__ =="__main__":
    start_time=time.time()
    retrieve_html()
    stop_time=time.time()
    print("Time taken {}".format(stop_time-start_time))
            
            
                
