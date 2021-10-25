
################################################Importing Packages###########################################
import fhirbase
import psycopg2
import pandas as pd
import ast
import json
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import warnings
warnings.filterwarnings('ignore')

start_time = time.time()
################################################Connecting to Postgres###########################################
##conn = psycopg2.connect(host="patientory.cnyrvm7s6vwa.us-east-1.rds.amazonaws.com",database="fhirbase",user="postgres",password="i02D7Jj1mWiLfY2MNsya")
conn = psycopg2.connect(host="localhost",database="fhirbase",user="postgres",password="postgres")
print("Connection with Postgres established")
##############################################################################################################
################################################Data Management###############################################
##############################################################################################################


############ Patient Data##########
print("Importing Patient Data")

patient = pd.read_sql_query('''Select
                                   resource#>>'{id}' as PatientID,
                                   resource#>>'{name,0,given,0}' as  PatientFirstName,
                                   resource#>>'{name,0,family}' as  PatientLastName,
                                   resource#>>'{name,0,prefix,0}' as  Prefix,
                                   resource#>>'{name,0,use}' as  NameType,
                                   resource#>>'{text,div}' as  Text,
                                   resource#>>'{text,status}' as  Status,
                                   resource#>>'{gender}' as  Gender,
                                   resource#>>'{address,0,line,0}' as  Street,
                                   resource#>>'{address,0,city}' as  City,
                                   resource#>>'{address,0,state}' as  State,
                                   resource#>>'{address,0,country}' as  Country,
                                   resource#>>'{address,0,postalCode}' as  Zipcode,
                                   resource#>>'{address,0,extension,0,extension,0,valueDecimal}' as  Latitude,
                                   resource#>>'{address,0,extension,0,extension,1,valueDecimal}' as  Longitude,
                                   resource#>>'{telecom,0,use}' as  TelecomUse,
                                   resource#>>'{telecom,0,value}' as  TelecomNumber,
                                   resource#>>'{birthDate}' as  DOB,
                                   resource#>>'{identifier}' as  Identifier,
                                   resource#>>'{resourceType}' as  ResourceType,
                                   resource#>>'{maritalStatus,coding,0,code}' as  MaritalStatus,
                                   resource#>>'{multipleBirth,boolean}' as  MultipleBirth
                            from patient''',con=conn)

print('#')
print("#")
print("#")
print("Imported data from patient table. Total {} records".format(len(patient)))


#######Breaking identifier column in patient data####################
print("Breaking identifier column from patient data")

identifier_col = list(patient['identifier'])
final_identifier = pd.DataFrame()
for identifier in identifier_col:
    s_id = ast.literal_eval(identifier) # identifier shows up as a string. so convert it to type dict
    df_identifier = pd.json_normalize(s_id)
    df_identifier = df_identifier[['type.text', 'value']]
    df_identifier['type.text'] = df_identifier['type.text'].apply(lambda x: 'patientid' if pd.isnull(x) else x)
    df_identifier['patientid'] = df_identifier['value'].loc[df_identifier['type.text']== 'patientid'].unique()[0]
    final_identifier = final_identifier.append(df_identifier)
    
final_identifier = final_identifier.pivot_table(values='value', index='patientid', columns='type.text', aggfunc='first')
final_identifier.drop('patientid', inplace = True, axis = 1)
final_identifier.reset_index(inplace= True)
patient = patient.merge(final_identifier, how = 'left', on ='patientid')
patient.drop(columns=['identifier','text'],inplace = True)

##############Calculating age from DOB###############
print("Calculating age using Date of Birth")
patient['dob'] = patient['dob'].astype('datetime64[ns]')
now = pd.Timestamp('now')
patient['dob'] = pd.to_datetime(patient['dob'], format='%m%d%y')    # 1
patient['dob'] = patient['dob'].where(patient['dob'] < now, patient['dob'] -  np.timedelta64(100, 'Y'))   # 2
patient['age'] = (now - patient['dob']).astype('<m8[Y]')    # 3
patient['age'] = patient['age'].astype('int')

print('#')
print("#")
print("#")
print("Completed cleaning Patient data")

#########Condition Data##############################
print("Importing Condition Data")
condition = pd.read_sql_query('''Select
                                       resource#>>'{id}' as  ConditionID,
                                       resource#>>'{code,text}' as  Disease,
                                       resource#>>'{code,coding,0,code}' as  Code,
                                       resource#>>'{code,coding,0,display}' as  DiseaseName,
                                       resource#>>'{onset,dateTime}' as  DateTime,
                                       resource#>>'{subject,id}' as  PatientID,
                                       resource#>>'{subject,resourceType}' as  ResourceType,
                                       resource#>>'{clinicalStatus,coding,0,code}' as  clinicalStatusCode,
                                       resource#>>'{verificationStatus,coding,0,code}' as  verificationStatusCode
                                from condition''',con=conn)

print('#')
print("#")
print("#")
print("Imported data from condition table. Total {} records".format(len(condition)))

condition = pd.read_sql_query('''select * from condition''',con=conn)


#########Observation Data##############################
print("Importing Observation Data for Kidney Disease")
Kidney_Observations = ('Calcium','Microalbumin Creatinine Ratio','Estimated Glomerular Filtration Rate'
                       ,'Blood Pressure','Respiratory rate','Urea Nitrogen','Creatinine'
                       ,'Sodium','Potassium','Chloride','Glomerular filtration rate/1.73 sq M.predicted'
                       ,'Globulin [Mass/volume] in Serum by calculation','Hemoglobin [Mass/volume] in Blood'
                       ,'Urea nitrogen [Mass/volume] in Serum or Plasma','Creatinine [Mass/volume] in Serum or Plasma'
                       ,'Calcium [Mass/volume] in Serum or Plasma','Sodium [Moles/volume] in Serum or Plasma'
                       ,'Potassium [Moles/volume] in Serum or Plasma','Chloride [Moles/volume] in Serum or Plasma'
                       ,'NT-proBNP','Appearance of Urine','Appearance of Urine')




observation = pd.read_sql_query('''Select
                                           resource#>>'{id}' as  ObservationID,
                                           resource#>>'{code,text}' as  Observation,
                                           resource#>>'{code,coding,0,code}' as  ObservationCode,
                                           resource#>>'{code,coding,0,display}' as  ObservationDisplay,
                                           resource#>>'{issued}' as  DateIssued,
                                           resource#>>'{status}' as  Status,
                                           resource#>>'{subject,id}' as  PatientId,
                                           resource#>>'{subject,resourceType}' as  obsresourceType,
                                           resource#>>'{component}' as  component,
                                           resource#>>'{category,0,coding,0,code}' as  CategoryCode,
                                           resource#>>'{value,Quantity,value}' as Quantity
                                    from observation
                                    where resource#>>'{code,text}' in ('Calcium','Microalbumin Creatinine Ratio','Estimated Glomerular Filtration Rate'
                                                           ,'Respiratory rate','Urea Nitrogen','Creatinine'
                                                           ,'Sodium','Potassium','Chloride','Glomerular filtration rate/1.73 sq M.predicted'
                                                           ,'Globulin [Mass/volume] in Serum by calculation','Hemoglobin [Mass/volume] in Blood'
                                                           ,'Urea nitrogen [Mass/volume] in Serum or Plasma','Creatinine [Mass/volume] in Serum or Plasma'
                                                           ,'Calcium [Mass/volume] in Serum or Plasma','Sodium [Moles/volume] in Serum or Plasma'
                                                           ,'Potassium [Moles/volume] in Serum or Plasma','Chloride [Moles/volume] in Serum or Plasma'
                                                           ,'NT-proBNP','Appearance of Urine','Appearance of Urine')
                                    order by resource#>>'{subject,id}';''',con=conn)
observation['ObservationName'] = observation['observationdisplay'].apply(lambda x: 'obs_' + x)


observation_pivot = observation.loc[:,['patientid','ObservationName','quantity']]
observation_pivot['quantity'] = observation_pivot['quantity'].astype(float)
#observation_pivot['has_obs'] = 1
observation_pivot = pd.pivot_table(observation_pivot, values='quantity', index='patientid',columns='ObservationName', aggfunc='mean')

print('#')
print("#")
print("#")
print("Completed importing and pivoting the Observaton data with kidney disease.. Total {} records".format(len(observation)))

#########Careplan Data##############################

careplan = pd.read_sql_query('''Select
                                           resource#>>'{id}' as  CareplanId,
                                           resource#>>'{meta,profile,0}' as  ProfileId,
                                           resource#>>'{intent}' as  Intent,
                                           resource#>>'{period,start}' as  StartDate,
                                           resource#>>'{period,end}' as  EndDate,
                                           resource#>>'{status}' as  Status,
                                           resource#>>'{subject,id}' as  PatientId,
                                           resource#>>'{subject,resourceType}' as  ResourceType,
                                           resource#>>'{activity,0,detail,code,text}' as  Text1,
                                           resource#>>'{activity,1,detail,code,text}' as  Text2,
                                           resource#>>'{activity,2,detail,code,text}' as  Text3,
                                           resource#>>'{activity,3,detail,code,text}' as  Text4,
                                           resource#>>'{activity,5,detail,code,text}' as  Text5,
                                           resource#>>'{activity,0,detail,status}' as  activitystatus1,
                                           resource#>>'{activity,1,detail,status}' as  activitystatus2,
                                           resource#>>'{activity,2,detail,status}' as  activitystatus3,
                                           resource#>>'{activity,3,detail,status}' as  activitystatus4,
                                           resource#>>'{activity,4,detail,status}' as  activitystatus5,
                                           resource#>>'{careTeam,0,id}' as  CareTeamId
                                    from careplan''',con=conn)


#############Pulling only relevant columns from careplan table####
careplan_text1 = careplan.loc[:,['patientid','intent','status','text1']]
careplan_text1 = careplan_text1.rename(columns = {'text1': 'careplanname'})
careplan_text2 = careplan.loc[:,['patientid','intent','status','text2']]
careplan_text2 = careplan_text2.rename(columns = {'text2': 'careplanname'})
careplan_text3 = careplan.loc[:,['patientid','intent','status','text3']]
careplan_text3 = careplan_text3.rename(columns = {'text3': 'careplanname'})
careplan_text4 = careplan.loc[:,['patientid','intent','status','text4']]
careplan_text4 = careplan_text4.rename(columns = {'text4': 'careplanname'})
careplan_text5 = careplan.loc[:,['patientid','intent','status','text5']]
careplan_text5 = careplan_text5.rename(columns = {'text5': 'careplanname'})
careplan_final = pd.concat([careplan_text1,careplan_text2,careplan_text3,careplan_text4,careplan_text5])
careplan_final = careplan_final.loc[careplan_final['careplanname'].notnull(),: ]
careplan_kidney = careplan_final.loc[careplan_final['careplanname'].isin(['low salt diet education','Low salt diet education (procedure)'
                                                                               ,'Low sodium diet (finding)','Administration of intravenous fluids',
                                                                               'Alcohol-free diet','Urine screening']),:]
careplan_kidney['careplanname'] = careplan_kidney['careplanname'].apply(lambda x: 'cp_'+ x)
careplan_pivot = careplan_kidney.loc[:,['patientid','careplanname']]
careplan_pivot['has_careplan'] = 1
careplan_pivot = pd.pivot_table(careplan_pivot, values='has_careplan', index='patientid',columns='careplanname', aggfunc='sum')
careplan_pivot = careplan_pivot.fillna(0)


#########Procedure Data##############################

procedure = pd.read_sql_query('''Select
                                       resource#>>'{id}' as  prodedureid,
                                       resource#>>'{code,text}' as  ProcedureName,
                                       resource#>>'{code,coding,0,code}' as  ProcedureCode,
                                       resource#>>'{status}' as  Procedurestatus,
                                       resource#>>'{subject,id}' as  PatientId,
                                       resource#>>'{performed,Period,start}' as  ProcedureStartDate,
                                       resource#>>'{performed,Period,end}' as  ProcedureEndDate
                                from procedure''',con=conn)

procedure_kidney = procedure.loc[procedure['procedurename'].isin(['Manual pelvic examination (procedure)',
                                                                 'Sputum examination (procedure)',
                                                                 'RhD passive immunization',
                                                                 'Gonorrhea infection test']),:]

procedure_kidney['procedurename'] = procedure_kidney['procedurename'].apply(lambda x: 'proc_'+ x)

####Pivoting procedure to remove duplicates#########
procedure_pivot = procedure_kidney.loc[:,['patientid','procedurename']]
procedure_pivot['completed_procedure'] = 1
procedure_pivot = pd.pivot_table(procedure_pivot, values='completed_procedure', index='patientid',columns='procedurename', aggfunc='sum')
procedure_pivot = procedure_pivot.fillna(0)



##############Merging patient with condition and pulling kidney disease###########
df = pd.merge(patient,condition,on = 'patientid',how = 'left')
df_kidney_patient_cond = df.loc[df['diseasename'].str.contains('kidney' , na = False),:]
df_kidney_patient_cond.drop(columns=['conditionid','disease','code','datetime','resourcetype_x','resourcetype_y'],inplace = True)


###########Merging patient, condition, observation,procedure and careplan data################
df_kidney_pat_cond_cp = pd.merge(df_kidney_patient_cond,careplan_pivot,how = 'left',on = 'patientid')
df_kidney_pat_cond_cp.drop_duplicates()
df_kidney_pat_cond_cp = df_kidney_pat_cond_cp.fillna(0)
df_kidney_pat_cond_cp_proc = pd.merge(df_kidney_pat_cond_cp,procedure_pivot,how = 'left',on = 'patientid')
df_kidney_pat_cond_cp_proc.drop_duplicates()
df_kidney_pat_cond_cp_proc = df_kidney_pat_cond_cp_proc.fillna(0)
df_kidney_pat_cond_cp_proc_obs = pd.merge(df_kidney_pat_cond_cp_proc,observation_pivot,how = 'left',on = 'patientid')
df_kidney_pat_cond_cp_proc_obs.drop_duplicates()
#df_kidney_pat_cond_cp_proc_obs = df_kidney_pat_cond_cp_proc_obs.fillna(0.0)
df_kidney_final = df_kidney_pat_cond_cp_proc_obs.copy()
df_kidney_final = pd.get_dummies(df_kidney_final, columns=['gender'])


#################################################################################################################
####################################################Model Analysis################################################
################################################################################################################


#############Feature Selection##########

patient_variables = ['gender_male','gender_female','age']

careplan_variables = ['cp_Administration of intravenous fluids','cp_Alcohol-free diet'
                      ,'cp_Low salt diet education (procedure)'
                      ,'cp_Low sodium diet (finding)',	'cp_Urine screening','cp_low salt diet education']

procedure_varibales = ['proc_Gonorrhea infection test','proc_Manual pelvic examination (procedure)'
                       ,'proc_RhD passive immunization','proc_Sputum examination (procedure)']

observation_variables = ["obs_Calcium","obs_Calcium [Mass/volume] in Serum or Plasma"
                         ,"obs_Chloride","obs_Chloride [Moles/volume] in Serum or Plasma","obs_Creatinine"
                         ,"obs_Creatinine [Mass/volume] in Serum or Plasma","obs_Estimated Glomerular Filtration Rate"
                         ,"obs_Globulin [Mass/volume] in Serum by calculation"
                         ,"obs_Glomerular filtration rate/1.73 sq M.predicted","obs_Hemoglobin [Mass/volume] in Blood"
                         ,"obs_Microalbumin Creatinine Ratio","obs_NT-proBNP","obs_Potassium"
                         ,"obs_Potassium [Moles/volume] in Serum or Plasma","obs_Respiratory rate"
                         ,"obs_Sodium","obs_Sodium [Moles/volume] in Serum or Plasma","obs_Urea Nitrogen"
                         ,"obs_Urea nitrogen [Mass/volume] in Serum or Plasma"]

impute_it = IterativeImputer()
impute_pat_obs_final = df_kidney_final.loc[:,patient_variables].values
for obs in observation_variables:
    obs_pat_feature = patient_variables.copy()
    obs_pat_feature.append(obs) 
    df_obs_pat_feature = df_kidney_final.loc[:,obs_pat_feature]
    impute_pat_obs = impute_it.fit_transform(df_obs_pat_feature)
    impute_pat_obs = impute_pat_obs[:,3]
    impute_pat_obs_final = np.column_stack((impute_pat_obs_final,impute_pat_obs))
impute_pat_obs_final 




X = df_kidney_final.loc[:,patient_variables + careplan_variables + procedure_varibales + observation_variables]
y = df_kidney_final.loc[:,'diseasename']
df_kidney_model = df_kidney_final.loc[:,patient_variables + careplan_variables + procedure_varibales + observation_variables+ ['diseasename']]

######################Visualizations###############################
df_kidney_model['diseasename'].value_counts().plot(kind='bar');

#Univariate analysis age.
f = plt.figure(figsize=(20,4))
f.add_subplot(1,2,1)
sns.distplot(df_kidney_model['age'])

f.add_subplot(1,2,2)
sns.boxplot(df_kidney_model['age'])

#Univariate analysis sex: 1=male; 0=female.
sns.countplot(df_kidney_model['gender_male'])

######################Hyperparameter Tuning for Model##############

model_params = {
    'knn': {
        'model': KNeighborsClassifier(),
        'params' : {
            'leaf_size': list(range(1,50)),
            'n_neighbors' : list(range(1,30)),
            'p': [1,2]
        }  
    }, 
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': list(range(1,25))
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
    'naive_bayes_multinomial': {
        'model': MultinomialNB(),
        'params': {}
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    },
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': list(range(1,5)),
            'kernel': ['rbf','linear']
        }  
    },
}



scores = []

for model_name, mp in model_params.items():
    print('running ' + model_name)
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
model_results = pd.DataFrame(scores,columns=['model','best_score','best_params'])


###########exporting data to excel################

##observation
observation.to_excel('observation.xlsx',index = False)
observation_pivot.to_excel('observation_pivot.xlsx',index = False)

##careplan
careplan_kidney.to_excel('careplan.xlsx',index = False)
careplan_pivot.to_excel('careplan_pivot.xlsx',index = False)

##procedure
procedure_kidney.to_excel('procedure.xlsx',index = False)
procedure_pivot.to_excel('procedure_pivot.xlsx',index = False)

##patient
patient.to_excel('patient.xlsx',index = False)

##condition
condition.to_excel('condition.xlsx',index = False)

##Joined
df_kidney_patient_cond.to_excel('df_kidney_pat_con.xlsx',index = False)
df_kidney_pat_cond_cp.to_excel('df_kidney_pat_cond_cp.xlsx',index = False)
df_kidney_pat_cond_cp_proc.to_excel('df_kidney_pat_cond_cp_proc.xlsx', index= False)
df_kidney_final.to_excel('df_kidney_final.xlsx', index= False)

##Model dataset
df_kidney_model.to_excel('df_kidney_model.xlsx', index= False)

##Model results
model_results.to_excel("model_results.xlsx", index = False)


elapsed_time = (time.time() - start_time)/60
print('Time taken to run this code {} mins'.format(elapsed_time))