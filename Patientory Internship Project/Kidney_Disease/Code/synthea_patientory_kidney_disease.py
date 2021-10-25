#!/usr/bin/env python
# coding: utf-8

# # Importing Packages

# In[1]:


#import fhirbase
import psycopg2
import random 
import pandas as pd
import ast
import json
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn import svm
from sklearn.ensemble  import GradientBoostingClassifier
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
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report



import warnings
warnings.filterwarnings('ignore')
random.seed(100000)
start_time = time.time()


# # Connecting to Postgres Database

# In[2]:


print("Connecting to Postgres database")
conn = psycopg2.connect(host="localhost",database="ptoy",user="postgres",password="postgres")


# # Data Management

# ### Patient Table Data

# In[3]:


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
                                   left(resource#>>'{deceased,dateTime}',10) as DeceasedDate,
                                   resource#>>'{identifier}' as  Identifier,
                                   resource#>>'{resourceType}' as  ResourceType,
                                   resource#>>'{maritalStatus,coding,0,code}' as  MaritalStatus,
                                   resource#>>'{multipleBirth,boolean}' as  MultipleBirth
                            from patient''',con=conn)

print("Imported data from patient table. Total {} records".format(len(patient)))


# ### Cleaning Patient Data

# In[4]:


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
patient['deceaseddate'] = patient['deceaseddate'].astype('datetime64[ns]')
now = pd.Timestamp('now')
patient['dob'] = pd.to_datetime(patient['dob'], format='%m%d%y')    # 1
patient['deceaseddate'] = pd.to_datetime(patient['deceaseddate'], format='%m%d%y')
patient['dob'] = patient['dob'].where(patient['dob'] < now, patient['dob'] -  np.timedelta64(100, 'Y'))   # 2
patient['age'] = np.where(np.isnat(patient['deceaseddate']), (now - patient['dob']).astype('<m8[Y]') , (patient['deceaseddate'] - patient['dob']).astype('<m8[Y]'))
#patient['age'] = (now - patient['dob']).astype('<m8[Y]')    # 3
#patient['age'] = patient['age'].astype('int')

print("Completed cleaning Patient data")
patient.head(5)


# ### Condition Data

# In[5]:


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

print("Imported data from condition table. Total {} records".format(len(condition)))
condition.head(5)


# ### Observation Data

# In[6]:


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
                                           resource#>>'{value,Quantity,value}' as Quantity,
                                           resource#>>'{component,0,value,value}' as Dialostilic_Pressure,
                                           resource#>>'{component,1,value,value}' as Systolic_Pressure
                                    from observation
                                    where resource#>>'{code,text}' in ('Calcium','Microalbumin Creatinine Ratio','Estimated Glomerular Filtration Rate'
                                                           ,'Respiratory rate','Urea Nitrogen','Creatinine'
                                                           ,'Sodium','Potassium','Chloride','Glomerular filtration rate/1.73 sq M.predicted'
                                                           ,'Globulin [Mass/volume] in Serum by calculation','Hemoglobin [Mass/volume] in Blood'
                                                           ,'Urea nitrogen [Mass/volume] in Serum or Plasma','Creatinine [Mass/volume] in Serum or Plasma'
                                                           ,'Calcium [Mass/volume] in Serum or Plasma','Sodium [Moles/volume] in Serum or Plasma'
                                                           ,'Potassium [Moles/volume] in Serum or Plasma','Chloride [Moles/volume] in Serum or Plasma'
                                                           ,'NT-proBNP','Appearance of Urine','Appearance of Urine','Blood Pressure')
                                    order by resource#>>'{issued}';''',con=conn)
observation['ObservationName'] = observation['observationdisplay'].apply(lambda x: 'obs_' + x)

observation.sort_values(by=['patientid','dateissued'])
observation_pivot = observation.loc[:,['patientid','ObservationName','quantity']]
observation_pivot['quantity'] = observation_pivot['quantity'].astype(float)
#observation_pivot['has_obs'] = 1
observation_pivot = pd.pivot_table(observation_pivot, values='quantity', index='patientid',columns='ObservationName', aggfunc='last')
print("Completed importing and pivoting the Observaton data with kidney disease.. Total {} records".format(len(observation)))
observation.head(5)


# ### Careplan Data

# In[7]:


print("Importing Careplan Data for Kidney Disease")
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

##Pulling Careplans which are only relevant to Kidney Disease
careplan_kidney = careplan_final.loc[careplan_final['careplanname'].isin(['low salt diet education','Low salt diet education (procedure)'
                                                                               ,'Low sodium diet (finding)','Administration of intravenous fluids',
                                                                               'Alcohol-free diet','Urine screening']),:]
careplan_kidney['careplanname'] = careplan_kidney['careplanname'].apply(lambda x: 'cp_'+ x)

##Pivoting Careplan data
careplan_pivot = careplan_kidney.loc[:,['patientid','careplanname']]
careplan_pivot['has_careplan'] = 1
careplan_pivot = pd.pivot_table(careplan_pivot, values='has_careplan', index='patientid',columns='careplanname', aggfunc='sum')
careplan_pivot = careplan_pivot.fillna(0)

print("Imported data from Careplan table. Total {} records".format(len(careplan_kidney)))

careplan_pivot.head(5)


# ### Procedure Data

# In[8]:


print("Importing Procedure Data for Kidney Disease")
procedure = pd.read_sql_query('''Select
                                       resource#>>'{id}' as  prodedureid,
                                       resource#>>'{code,text}' as  ProcedureName,
                                       resource#>>'{code,coding,0,code}' as  ProcedureCode,
                                       resource#>>'{status}' as  Procedurestatus,
                                       resource#>>'{subject,id}' as  PatientId,
                                       resource#>>'{performed,Period,start}' as  ProcedureStartDate,
                                       resource#>>'{performed,Period,end}' as  ProcedureEndDate
                                from procedure''',con=conn)

##Pulling Procedures which are only relevant to Kidney Disease
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

print("Imported data from Procedure table. Total {} records".format(len(procedure_kidney)))

procedure_pivot.head(5)


# ### Merging patient with condition and pulling kidney disease

# In[9]:


df_kidney_patient_cond = pd.merge(patient,condition,on = 'patientid',how = 'left')
df_kidney_patient_cond = df_kidney_patient_cond.loc[df_kidney_patient_cond['diseasename'].str.contains('kidney' , na = False),:]
df_kidney_patient_cond.drop(columns=['conditionid','disease','code','datetime','resourcetype_x','resourcetype_y'],inplace = True)
print("Completed Merging Patient with Condition table and filtering only Kidney Disease. Total {} records".format(len(df_kidney_patient_cond)))
df_kidney_patient_cond.head(5)


# ### Merging patient, condition, observation,procedure and careplan data

# In[10]:


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
print("Completed Merging Patient-Condition table with Procedure, Careplan and Observation tables. Total {} records".format(len(df_kidney_final)))
df_kidney_final.head(5)


# # Model Creation and Analysis 

# ### Feature Selection

# In[11]:


print('Defining Features')

patient_variables = ['gender_male','gender_female','age']

careplan_variables = ['cp_Administration of intravenous fluids','cp_Alcohol-free diet'
                      ,'cp_Low salt diet education (procedure)'
                      ,'cp_Low sodium diet (finding)','cp_Urine screening','cp_low salt diet education']

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

print('Completed defining Features')


# ### Checking for Multicollinearity

# ### Imputing Missing Values

# In[12]:


print('Imputing missing values')
impute_it = IterativeImputer()
impute_pat_obs_final = df_kidney_final.loc[:,patient_variables].values
for obs in observation_variables:
    obs_pat_feature = patient_variables.copy()
    obs_pat_feature.append(obs) 
    df_obs_pat_feature = df_kidney_final.loc[:,obs_pat_feature]
    impute_pat_obs = impute_it.fit_transform(df_obs_pat_feature)
    impute_pat_obs = impute_pat_obs[:,3]
    impute_pat_obs_final = np.column_stack((impute_pat_obs_final,impute_pat_obs))

print('Finished imputing missing values for observations')

impute_pat_obs_final_columns = patient_variables + observation_variables
df_impute_pat_obs_final = pd.DataFrame(data = impute_pat_obs_final, columns = impute_pat_obs_final_columns)


# In[13]:


df_vif = pd.DataFrame()
df_vif["VIF Factor"] = [variance_inflation_factor(df_impute_pat_obs_final.values, i) for i in range(df_impute_pat_obs_final.shape[1])]
df_vif["features"] = df_impute_pat_obs_final.columns
df_vif


# ### Defining Predictors and Response (X,y)

# In[14]:


X = impute_pat_obs_final
#X = np.column_stack((impute_pat_obs_final , df_kidney_final[careplan_variables].values , df_kidney_final[procedure_varibales].values))
y = df_kidney_final.loc[:,'diseasename']
df_kidney_model = df_kidney_final.loc[:,patient_variables + observation_variables + ['diseasename']]


# ### Visualizations

# In[15]:


#https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.
    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.        spacing (int): The distance between the labels and the bars.
    """
    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'
        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'
        # Use Y value as label and format number with one decimal place
        label = "{:.0f}".format(y_value)
        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            #rotation= 90,
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# ### Analysis of Response Variable

# In[16]:


ax = df_kidney_model['diseasename'].value_counts().plot(kind='bar', figsize = (9,6), width = 0.5);
plt.title("Total Patients Per Kidney Disease")
plt.xlabel("Kidney Disease")
plt.ylabel("Total Patients")
ax = add_value_labels(ax)
plt.show()
plt.clf()


# ### Univariate analysis for Age (dist plot)

# In[17]:


f = plt.figure(figsize=(20,4))
f.add_subplot(1,2,1)
sns.distplot(df_kidney_model['age'])


# ### Univariate analysis for Age (Box Plot)

# In[18]:


f.add_subplot(1,2,2)
sns.boxplot(df_kidney_model['age'])


# ### Univariate analysis sex: 1=male; 0=female.

# In[19]:


ax = df_kidney_model['gender_male'].value_counts().plot(kind='bar', figsize = (9,6), width = 0.7);
plt.title("Total Patients by Gender")
plt.xlabel("Gender")
plt.ylabel("Total Patients")
ax = add_value_labels(ax)
plt.show()
plt.clf()


# ### Scaling of Data

# In[20]:


print("Scaling the Features")
sc_X = StandardScaler()
X_Scale = sc_X.fit_transform(X)
print('Completed scaling dependent features')


# ### Hyperparameter Tuning for Model

# In[21]:


print("Running KNN, SVM, Random Forest and Decision Tree Model")

model_params = {
    'KNN': {
        'model': KNeighborsClassifier(),
        'params' : {
            'leaf_size': list(range(1,50)),
            'n_neighbors' : list(range(1,30)),
            'p': [1,2]
        }  
    }, 
    'Logistic Regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': list(range(1,10))
        }
    },  
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': list(range(1,50))
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],  
            'splitter' : ['best','random'],
            'min_samples_leaf' : list(range(1,50))
        }
    },
    'SVM': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': list(range(1,20)),
            'kernel': ['rbf','linear']
        }  
    },
}



scores = []

for model_name, mp in model_params.items():
    print('Running ' + model_name)
    clf =  GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(X_Scale, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
hyperparameter_tuning_results = pd.DataFrame(scores,columns=['model','best_score','best_params'])

print('Completed Generating Model Results')

hyperparameter_tuning_results.head(10)


# ## Evaluating Each Results

# ### Splitting Datasets

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X_Scale, y, test_size=0.20, random_state = 550)
print("Datasets splitted to 80-20")


# ### KNN Classifier Analysis

# In[23]:


KNN_Model = KNeighborsClassifier(n_neighbors=19,leaf_size =1,p=1)
KNN_Model.fit(X_train, y_train)
print("Defined and fit the classifier")
y_pred = KNN_Model.predict(X_test)
print("Predicted the output on test data")
print("Accuracy:",KNN_Model.score(X_test, y_test))

labels = list(set(y_train))
labels.sort()
knn_cm = confusion_matrix(y_test,y_pred, labels = labels)
ax= plt.subplot()
sns.heatmap(knn_cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted');
ax.set_ylabel('Actual'); 
ax.set_title('Confusion matrix of the KNN Classifier'); 
ax.xaxis.set_ticklabels(labels,rotation = 90); 
ax.yaxis.set_ticklabels(labels,rotation = 0);


# ### Logistic Regression Analysis

# In[35]:


Logistic_Model = LogisticRegression(solver='liblinear',multi_class='auto',C=3)
clf  = Logistic_Model.fit(X_train, y_train)
print("Defined and fit the classifier")
y_pred = Logistic_Model.predict(X_test)
print("Predicted the output on test data")
print("Accuracy:",Logistic_Model.score(X_test, y_test))

df_y_pred = pd.DataFrame(y_pred)
df_y_pred = df_y_pred.rename(columns={0:''})
df_y_pred = pd.get_dummies(df_y_pred)

estimates = pd.DataFrame(clf.coef_.T)

estimates = estimates.rename(columns = {0: 'Chronic kidney disease stage 1 (disorder) Estimates',
                                       1: 'Chronic kidney disease stage 2 (disorder) Estimates',
                                       2: 'Chronic kidney disease stage 3 (disorder) Estimates',
                                       3: 'Injury of kidney (disorder) Estimates'})

features = list(df_kidney_final.loc[:,patient_variables + observation_variables])
features = pd.DataFrame(features)
features = features.rename(columns = {0:'Features'})
df_logit_Model_resuts = pd.concat([features, estimates], axis=1)
intercept_values = list(clf.intercept_)
intercept = ['intercept']
intercept.extend(intercept_values)
df_logit_Model_resuts.loc[len(df_logit_Model_resuts), :] = intercept
df_logit_Model_resuts


# In[52]:


df_logit_Model_resuts.to_excel('df_kidney_model.xlsx', index= False)


# In[36]:


labels = list(set(y_train))
labels.sort()
lr_cm = confusion_matrix(y_test,y_pred, labels = labels)
ax= plt.subplot()
sns.heatmap(lr_cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
ax.set_title('Confusion matrix of the Logistic Regressor'); 
ax.xaxis.set_ticklabels(labels,rotation = 90); 
ax.yaxis.set_ticklabels(labels,rotation = 0);


# ### Decision Tree Classifier Analysis 

# In[47]:


from sklearn import tree
DT_Model = tree.DecisionTreeClassifier(criterion = "gini",splitter = 'best',min_samples_leaf = 25) 
DT_CLF = DT_Model.fit(X_train, y_train)
print("Defined and fit the classifier")
plt.figure(figsize = (20,15))
tree.plot_tree(DT_CLF, filled = True)


# In[38]:



print(tree.export_text(DT_CLF))


# In[48]:


y_pred = DT_Model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

labels = list(set(y_train))
labels.sort()
dt_cm = confusion_matrix(y_test,y_pred, labels = labels)
ax= plt.subplot()
sns.heatmap(dt_cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
ax.set_title('Confusion matrix of the Decision Tree'); 
ax.xaxis.set_ticklabels(labels,rotation = 90); 
ax.yaxis.set_ticklabels(labels,rotation = 0);

print(classification_report(y_test, y_pred))

print("Accuracy:",DT_Model.score(X_test, y_test))


# In[50]:


#https://machinelearningmastery.com/calculate-feature-importance-with-python/
importance = DT_Model.feature_importances_
df_importance = pd.DataFrame(importance, columns=['importance_score'])
features_plot = pd.concat([features,df_importance], axis=1)
print(features_plot.sort_values('importance_score',ascending = False) )

features_plot.plot.bar(x='Features',y='importance_score')


# ### Random Forest

# In[44]:



RandomForest_Model = RandomForestClassifier(n_estimators = 45)
rf_clf  = RandomForest_Model.fit(X_train, y_train)
print("Defined and fit the classifier")
y_pred = RandomForest_Model.predict(X_test)
print("Predicted the output on test data")
print("Accuracy:",RandomForest_Model.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print("Accuracy:",RandomForest_Model.score(X_test, y_test))


labels = list(set(y_train))
labels.sort()
dt_cm = confusion_matrix(y_test,y_pred, labels = labels)
ax= plt.subplot()
sns.heatmap(dt_cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
ax.set_title('Confusion matrix of the Random Forest'); 
ax.xaxis.set_ticklabels(labels,rotation = 90); 
ax.yaxis.set_ticklabels(labels,rotation = 0);


# In[31]:


importance = RandomForest_Model.feature_importances_
df_importance = pd.DataFrame(importance, columns=['importance_score'])
features_plot = pd.concat([features,df_importance], axis=1)
print(features_plot.sort_values('importance_score',ascending = False) )

features_plot.plot.bar(x='Features',y='importance_score')


# ### SVM Analysis

# In[45]:


from sklearn import svm
#Create a svm Classifier
SVM_Model = svm.SVC(C = 3,kernel='linear') # Linear Kernel
#Train the model using the training sets
SVM_CLF = SVM_Model.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = SVM_Model.predict(X_test)

print("Predicted the output on test data")
print("Accuracy:",SVM_Model.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print("Accuracy:",SVM_Model.score(X_test, y_test))


labels = list(set(y_train))
labels.sort()
dt_cm = confusion_matrix(y_test,y_pred, labels = labels)
ax= plt.subplot()
sns.heatmap(dt_cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
ax.set_title('Confusion matrix of the Random Forest'); 
ax.xaxis.set_ticklabels(labels,rotation = 90); 
ax.yaxis.set_ticklabels(labels,rotation = 0);



estimates = pd.DataFrame(SVM_CLF.coef_.T)

estimates = estimates.rename(columns = {0: 'Chronic kidney disease stage 1 (disorder) Estimates',
                                       1: 'Chronic kidney disease stage 2 (disorder) Estimates',
                                       2: 'Chronic kidney disease stage 3 (disorder) Estimates',
                                       3: 'Injury of kidney (disorder) Estimates'})


# ### Exporting to Excel

# In[33]:



##observation
observation.to_excel('observation.xlsx',index = False)
observation_pivot.to_excel('observation_pivot.xlsx',index = False)
print('Exported Observation Data')

##careplan
careplan_kidney.to_excel('careplan.xlsx',index = False)
careplan_pivot.to_excel('careplan_pivot.xlsx',index = False)
print('Exported Careplan Data')

##procedure
procedure_kidney.to_excel('procedure.xlsx',index = False)
procedure_pivot.to_excel('procedure_pivot.xlsx',index = False)
print('Exported Procedure Data')

##patient
patient.to_excel('patient.xlsx',index = False)
print('Exported Patient Data.')

##condition
condition.to_excel('condition.xlsx',index = False)
print('Exported Condition Data.')

##Joined
df_kidney_patient_cond.to_excel('df_kidney_pat_con.xlsx',index = False)
df_kidney_pat_cond_cp.to_excel('df_kidney_pat_cond_cp.xlsx',index = False)
df_kidney_pat_cond_cp_proc.to_excel('df_kidney_pat_cond_cp_proc.xlsx', index= False)
df_kidney_final.to_excel('df_kidney_final.xlsx', index= False)
print('Exported all Joined Tables')

##Model dataset
df_kidney_model.to_excel('df_kidney_model.xlsx', index= False)
print('Exported the dataset used for Models.')

##Model results
hyperparameter_tuning_results.to_excel("hyperparameter_tuning_results.xlsx", index = False)
print('Exported Hyperparameter Tuning results.')


# # Time Management

# In[34]:


elapsed_time = (time.time() - start_time)/60
print('Time taken to run this code {} mins'.format(elapsed_time))

