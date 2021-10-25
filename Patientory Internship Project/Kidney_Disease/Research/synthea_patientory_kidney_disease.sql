Select
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
                                   resource#>>'{identifier,0}' as  Identifier,
                                   resource#>>'{resourceType}' as  ResourceType,
                                   resource#>>'{maritalStatus,coding,0,code}' as  MaritalStatus,
                                   resource#>>'{multipleBirth,boolean}' as  MultipleBirth
from patient limit 10

Select
       resource#>>'{id}' as  ConditionID,
       resource#>>'{code,text}' as  Disease,
       resource#>>'{code,coding,0,code}' as  Code,
       resource#>>'{code,coding,0,display}' as  DiseaseName,
       resource#>>'{onset,dateTime}' as  DateTime,
       resource#>>'{subject,id}' as  PatientID,
       resource#>>'{subject,resourceType}' as  ResourceType,
       resource#>>'{clinicalStatus,coding,0,code}' as  clinicalStatusCode,
       resource#>>'{verificationStatus,coding,0,code}' as  verificationStatusCode
from condition limit 20

Select * from observation;


    Select
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
                           ,'Blood Pressure','Respiratory rate','Urea Nitrogen','Creatinine'
                           ,'Sodium','Potassium','Chloride','Glomerular filtration rate/1.73 sq M.predicted'
                           ,'Globulin [Mass/volume] in Serum by calculation','Hemoglobin [Mass/volume] in Blood'
                           ,'Urea nitrogen [Mass/volume] in Serum or Plasma','Creatinine [Mass/volume] in Serum or Plasma'
                           ,'Calcium [Mass/volume] in Serum or Plasma','Sodium [Moles/volume] in Serum or Plasma'
                           ,'Potassium [Moles/volume] in Serum or Plasma','Chloride [Moles/volume] in Serum or Plasma'
                           ,'NT-proBNP','Appearance of Urine','Appearance of Urine');

Select
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
       resource#>>'{careTeam,0}' as  CareTeamId
from careplan


Select
       resource#>>'{id}' as  prodedureid,
       resource#>>'{code,text}' as  ProcedureName,
       resource#>>'{code,coding,0,code}' as  ProcedureCode,
       resource#>>'{status}' as  Procedurestatus,
       resource#>>'{subject,id}' as  PatientId,
       resource#>>'{performed,Period,start}' as  ProcedureStartDate,
       resource#>>'{performed,Period,end}' as  ProcedureEndDate
from procedure