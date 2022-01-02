USE [Fair_Lending]
GO

/****** Object:  StoredProcedure [dbo].[Uspload_MR_UW_Exception]    Script Date: 12/1/2021 12:53:17 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



--Execute [Fair_Lending_SB].[dbo].[Uspload_MR_UW_Exception] '2020-01-01' ,'2020-06-30'

CREATE PROCEDURE  [dbo].[Uspload_MR_UW_Exception]
(
	@StartDate Date, @EndDate Date
)
AS
BEGIN

--#### Test Procedure ####
--Declare @StartDate Date, @EndDate Date
--Set @StartDate = '2020-01-01'
--Set @EndDate   = '2020-06-30'
--#### Test Ends ####


IF OBJECT_ID('tempdb..#MR_UW_Exceptions_Data') IS NOT NULL DROP TABLE #MR_UW_Exceptions_Data
SELECT  
       [LOANNUMBER]                 = [LOANNUMBER_NEW]
      ,[ACTIONDATE]                 = [ACTIONDATE]
      ,[EXCEPTION_FLAG]				= [EXCEPTION_FLAG]
      ,[EXPANDED_AUTHORITY_FLAG]	= [EXPANDED_AUTHORITY_FLAG]
      ,[EXCEPTION_CAT_1]			= [EXCEPTION_CAT_1]
      ,[EXCEPTION_CAT_2]			= [EXCEPTION_CAT_2]
      ,[EXCEPTION_CAT_3]			= [EXCEPTION_CAT_3]
      ,[EXCEPTION_CAT_4]			= [EXCEPTION_CAT_4]
      ,[EXCEPTION_TYPE_1]			= [EXCEPTION_TYPE_1]
      ,[EXCEPTION_TYPE_2]			= [EXCEPTION_TYPE_2]
      ,[EXCEPTION_TYPE_3]			= [EXCEPTION_TYPE_3]
      ,[EXCEPTION_TYPE_4]			= [EXCEPTION_TYPE_4]
      ,[DATA_PERIOD]				= [DATA_PERIOD]
Into #MR_UW_Exceptions_Data
FROM [Fair_Lending].[dbo].[MR_UW_Exceptions_Data]
Where [DATA_PERIOD] between @StartDate and @EndDate


IF OBJECT_ID('tempdb..#FL_Mortgage_Raw') IS NOT NULL DROP TABLE #FL_Mortgage_Raw
Select 
   *
   ,BookedStatus = Case When ACTIONTAKEN = 1 Then 'Booked' ELse 'Not Booked' End
   ,AgencyStatus    = Case
					    When PLANDD  like '%FNMA%' Then 'Agency'
					    When PLANDD  like '%FHA%' Then 'Agency'
					    When PLANDD  like '%VA%' Then 'Agency'
					    When PLANDD  like '%USDA%' Then 'Agency'
					    When PLANDD  like '%FHLMC%' Then 'Agency' 
					    When PLANDD IN ('MHP','PHFA Keystone Home Loan – Conventional') Then 'Agency' 
					    Else 'Non-Agency'
					  End
   ,[RowNum]     = ROW_NUMBER ( ) OVER (  PARTITION BY LOANNUMBER  order by ACTIONDATE Asc)  
Into #FL_Mortgage_Raw
From [Fair_Lending].[dbo].[FL_Mortgage_Raw]            fl_raw with (Nolock)
Where ACTIONDATE between @StartDate and @EndDate 



IF OBJECT_ID('tempdb..#BasePopulation') IS NOT NULL DROP TABLE #BasePopulation
SELECT
       exc.*
      ,mr.[ACTIONTAKEN]
      ,mr.[APPADDR]
      ,mr.[APPAGE]
      ,mr.[APPCITY]
      ,mr.[APPCREDSCORE]
      ,mr.[APPDATE]
      ,mr.[APPETHINICITY]
      ,mr.[APPFIRSTNAME]
      ,mr.[APPGENDER]
      ,mr.[APPLASTNAME]
      ,mr.[APPMETHOD]
      ,mr.[APPMIDNAME]
      ,mr.[APPRACE1]
      ,mr.[APPRACE2]
      ,mr.[APPRACE3]
      ,mr.[APPRACE4]
      ,mr.[APPRACE5]
      ,mr.[APPSTATE]
      ,mr.[APPZIP]
      ,mr.[AREA]
      ,mr.[BERATIO]
      ,mr.[BR_PS]
      ,mr.[BR_SUPP]
      ,mr.[BRANCHCODE]
      ,mr.[BRANCHNAME]
      ,mr.[CENSUSTRACT]
      ,mr.[COAPPAGE]
      ,mr.[COAPPCREDSCORE]
      ,mr.[COAPPETHINICITY]
      ,mr.[COAPPFIRSTNAME]
      ,mr.[COAPPGENDER]
      ,mr.[COAPPLASTNAME]
      ,mr.[COAPPMIDNAME]
      ,mr.[COAPPRACE1]
      ,mr.[COAPPRACE2]
      ,mr.[COAPPRACE3]
      ,mr.[COAPPRACE4]
      ,mr.[COAPPRACE5]
      ,mr.[COUNTY]
      ,mr.[CRA_FLAG]
      ,mr.[CRA_TYPE]
      ,mr.[CURRCODE]
      ,mr.[DENIAL1]
      ,mr.[DENIAL2]
      ,mr.[DENIAL3]
      ,mr.[DIVISION]
      ,mr.[EXT_APR]
      ,mr.[FEE_ADJ]
      ,mr.[FRRATIO]
      ,mr.[HMDA_LOANPURPOSE]
      ,mr.[HMDA_LOANTYPE]
      ,mr.[HMDA_OCCUPANCY]
      ,mr.[LIEN_POS]
      ,mr.[LN_OFFICER_PS]
      ,mr.[LN_OFFICER_SUPP]
      ,mr.[LOAN_OFFICER_ID]
      ,mr.[LOANPURCHASER]
      ,mr.[LOANPURPOSE]
      ,mr.[LOANTERM]
      ,mr.[LOCKDAYSGUAR]
      ,mr.[LOCKINDATE]
      ,mr.[LTV]
      ,mr.[MLS_LOAN_TYPE]
      ,mr.[MSA]
      ,mr.[NOTERATE]
      ,mr.[PLANDD]
      ,mr.[PROGRAMCODE]
      ,mr.[RATE_EXTENDDAYS]
      ,mr.[RELOCKDATE]
      ,mr.[STATE]
      ,mr.[TOTAL_INCOME]
      ,mr.[TOTALORIGINATION]
      ,mr.[UNITS]
      ,mr.[INTERESTONLYFLAG]
      ,mr.[LPMI]
      ,mr.[CLTV]
      ,mr.[POINTSDISCOUNT]
      ,mr.[CONDO]
      ,mr.[LENDERFUNDEDBUYDOWN]
      ,mr.[MEDICALPROFESSIONALINDICATOR]
      ,mr.[MORTGAGEINSURANCEINDICATOR]
      ,mr.[CASH_BROUGHT_TO_CLOSING_AMT]
      ,mr.[CASH_TAKEN_FROM_CLOSING_AMT]
      ,mr.[PRICINGSUBSIDY]
      ,mr.[REQUESTED_AMOUNT]
      ,mr.[REQUESTED_TERM]
      ,mr.[DOWN_PAYMENT]
      ,mr.[MORTGAGE_PROCESSOR]
      ,mr.[EXCEPTIONS]
      ,mr.[CREDIT_HISTORY]
      ,mr.[MILITARYSTATUS]
      ,mr.[PROPADDR]
      ,mr.[PROPERTY_TYPE_CATEGORY]
	  ,mr.[RowNum]
	  ,RACEETH_NAME                  = Case 
	                                     When APPETHINICITY = 1 then 'Hispanic'
										 When APPETHINICITY = 2 and (Len(Ltrim(Rtrim(APPRACE2))) > '0' ) then 'Multiracial'
										 When APPETHINICITY = 2 and Ltrim(Rtrim(APPRACE1)) = '1' then 'American Indian or Alaskan'
										 When APPETHINICITY = 2 and Ltrim(Rtrim(APPRACE1)) in ('2','4') then 'Asian or Pacific Islander'
										 When APPETHINICITY = 2 and Ltrim(Rtrim(APPRACE1)) = '3' then 'African American'
										 When APPETHINICITY = 2 and Ltrim(Rtrim(APPRACE1)) = '5' then 'Non-Hispanic White'
										 Else 'Not Available'
									   End
	  ,CORACEETH_NAME                = Case 
	                                     When (COAPPLASTNAME is  null or Len(Ltrim(Rtrim(COAPPLASTNAME)))	 = '0')  Then 'No Coapplicant'  
	                                     When COAPPETHINICITY = 1 then 'Hispanic'
										 When COAPPETHINICITY = 2 and (Len(Ltrim(Rtrim(COAPPRACE2))) > '0')  then 'Multiracial'
										 When COAPPETHINICITY = 2 and Ltrim(Rtrim(COAPPRACE1)) = '1' then 'American Indian or Alaskan'
										 When COAPPETHINICITY = 2 and Ltrim(Rtrim(COAPPRACE1)) in ('2','4') then 'Asian or Pacific Islander'
										 When COAPPETHINICITY = 2 and Ltrim(Rtrim(COAPPRACE1)) = '3' then 'African American'
										 When COAPPETHINICITY = 2 and Ltrim(Rtrim(COAPPRACE1)) = '5' then 'Non-Hispanic White'
										 Else 'Not Available'
									   End
	  ,Coapp_ind                     = Case When COAPPLASTNAME is not null or Len(Ltrim(Rtrim(COAPPLASTNAME)))	 > '0' Then 1 Else 0 End
	  ,GENDER_NAME                   = Case 
	                                       When Ltrim(Rtrim(APPGender)) = '1' Then 'Male' 
										   When Ltrim(Rtrim(APPGender)) = '2' Then 'Female'
										   Else 'Not Available' 
									   End
	  ,COGENDER_NAME                 = Case 
	                                       When (COAPPLASTNAME is null or Len(Ltrim(Rtrim(COAPPLASTNAME)))	 = '0')  Then 'No Coapplicant'  
	                                       When Ltrim(Rtrim(COAPPGENDER)) = '1' Then 'Male' 
										   When Ltrim(Rtrim(COAPPGENDER)) = '1' Then 'Female' 
										   Else 'Not Available' 
									   End
	  ,AGE_NAME                      = Case When Ltrim(Rtrim(APPAGE)) >= '62' Then '>=62' Else '<62' End
	  ,COAGE_NAME                    = Case 
	                                       When (COAPPLASTNAME is null or Len(Ltrim(Rtrim(COAPPLASTNAME)))	 = '0')  Then 'No Coapplicant'  
	                                       When Ltrim(Rtrim(COAPPAGE)) >= '62' Then '>=62' 
										   Else '<62' 
									   End
Into #BasePopulation
FROM #MR_UW_Exceptions_Data exc
     inner join #FL_Mortgage_Raw mr on exc.LOANNUMBER = mr.LOANNUMBER and mr.RowNum = 1
Where mr.ACTIONTAKEN in (1,2,3) and AgencyStatus = 'Non-Agency'


IF OBJECT_ID('tempdb..#FinalBasePopulation') IS NOT NULL DROP TABLE #FinalBasePopulation
Select 
       * 
      ,EFF_RACE_NAME   = Case 
                           When (CORACEETH_NAME in ('No Coapplicant','Not Available') Or RACEETH_NAME in ('Hispanic','Multiracial','American Indian or Alaskan','Asian or Pacific Islander','African American')) Then RACEETH_NAME Else CORACEETH_NAME
	     			     End
      ,EFF_GENDER_NAME = Case When GENDER_NAME =  'Not Available' and Coapp_ind=1 Then COGENDER_NAME ELse  GENDER_NAME End
	  ,EFF_AGE_NAME    = Case	
						   When COAGE_NAME = 'No Coapplicant' then AGE_NAME
						   When AGE_NAME = '>=62' or COAGE_NAME = '>=62' then  '>=62'
						   Else '<62'
						 End
	  ,Is_Exception_Approved       = Case 
										When [EXCEPTION_FLAG] = 'Y' and ACTIONTAKEN in (1,2)    Then 1 
										When [EXCEPTION_FLAG] = 'Y' and ACTIONTAKEN in (3)    Then 0 
										Else Null
									 End
	  ,Is_Exception_Declined       = Case 
										When [EXCEPTION_FLAG] = 'Y' and ACTIONTAKEN in (1,2)    Then 0 
										When [EXCEPTION_FLAG] = 'Y' and ACTIONTAKEN in (3)    Then 1 
										Else Null
									 End
	  ,Is_Exception_Considered     = Case 
										When [EXCEPTION_FLAG] = 'Y'  Then 1
										When [EXCEPTION_FLAG] = 'N' and ACTIONTAKEN in (3)    Then 0 
										Else Null
									 End
	  ,Is_Exception_Not_Considered = Case 
										When [EXCEPTION_FLAG] = 'Y'  Then 0
										When [EXCEPTION_FLAG] = 'N' and ACTIONTAKEN in (3)    Then 1 
										Else Null
									 End
Into #FinalBasePopulation
From #BasePopulation


Truncate Table [Fair_Lending].dbo.MR_UW_Exceptions_Data_Analysis

Insert into [Fair_Lending].dbo.MR_UW_Exceptions_Data_Analysis
Select * from #FinalBasePopulation

End



GO


