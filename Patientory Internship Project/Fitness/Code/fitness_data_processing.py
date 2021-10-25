#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[1]:


import fhirbase
import psycopg2
import pandas as pd
import ast
import json
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
#import lux

import warnings
warnings.filterwarnings('ignore')

start_time = time.time()


# # Data Management

# ### Connecting to PostgreSQL

# In[2]:


print("Connecting to Postgres database")
##conn = psycopg2.connect(host="patientory.cnyrvm7s6vwa.us-east-1.rds.amazonaws.com",database="fhirbase",user="postgres",password="i02D7Jj1mWiLfY2MNsya")
conn = psycopg2.connect(host="localhost",database="fitness",user="postgres",password="postgres")
print("#")
print("#")
print("#")
print("Connection with Postgres established")


# ### Importing Fitness Table

# In[3]:


print("Importing Patient Data")

fitness = pd.read_sql_query('''select * from fitness''',conn)
prefix = 'HKQuantityTypeIdentifier'
len_prefix = len(prefix)
fitness['type'] = fitness['type'].apply(lambda x: x[len_prefix:])
fitness['End_Date'] = fitness['endDate'].apply(lambda x: x[:10])
grouped_fitness = fitness.groupby(['type', 'unit','End_Date']).agg({'value': ['mean']})
grouped_fitness = grouped_fitness.reset_index()

print('#')
print("#")
print("#")
print("Imported data from patient table. Total {} records".format(len(fitness)))
grouped_fitness.head(5)


# ### Importing Fitness CDA Table

# In[4]:


print("Importing Patient Data")

fitnessCDA = pd.read_sql_query('''select * from fitness_cda''',conn)


print('#')
print("#")
print("#")
print("Imported data from patient table. Total {} records".format(len(fitnessCDA)))
fitnessCDA.head(5)


# ### Exporting to Excel

# In[5]:


##fitness
fitness.to_excel('fitness.xlsx',index = False)
grouped_fitness.to_excel('grouped_fitness.xlsx')
print('Exported fitness Data')

##fitnessCDA
fitnessCDA.to_excel('fitnessCDA.xlsx',index = False)
print('Exported fitnessCDA Data')


# # Time Management

# In[6]:


elapsed_time = (time.time() - start_time)/60
print('Time taken to run this code {} mins'.format(elapsed_time))

