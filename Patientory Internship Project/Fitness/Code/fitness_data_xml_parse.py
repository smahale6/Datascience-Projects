#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import xml.etree.ElementTree as ET
import io

def iter_docs(HealthData,rcd):
    pat_attr = HealthData.attrib
    for doc in HealthData.iter(rcd):
        doc_dict = pat_attr.copy()
        doc_dict.update(doc.attrib)
        doc_dict['data'] = doc.text
        yield doc_dict

etree = ET.parse('export_1.xml') #create an ElementTree object 
df_health = pd.DataFrame(list(iter_docs(etree.getroot(),'Record')))
df_patient = pd.DataFrame(list(iter_docs(etree.getroot(),'ClinicalRecord')))


# In[9]:


df_health['patient'] = df_patient[df_patient['type']=='Patient']['resourceFilePath'].str.split('Patient-')[0][1][:-5]


# In[15]:


df_health[df_health['type'] == 'HKQuantityTypeIdentifierBloodGlucose']

