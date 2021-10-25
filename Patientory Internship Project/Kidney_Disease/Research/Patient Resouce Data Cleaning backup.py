import fhirbase
import psycopg2
import pandas as pd
import ast

conn = psycopg2.connect(host="patientory.cnyrvm7s6vwa.us-east-1.rds.amazonaws.com",database="fhirbase",user="postgres",password="i02D7Jj1mWiLfY2MNsya")

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


sample = patient.head(1)
identifier_col = list(sample['identifier'])
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


