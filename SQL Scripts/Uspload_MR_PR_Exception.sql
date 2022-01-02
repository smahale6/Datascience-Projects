USE [Fair_Lending]
GO

/****** Object:  StoredProcedure [dbo].[Uspload_MR_PR_Exception]    Script Date: 12/1/2021 12:54:22 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



--Execute [Fair_Lending].[dbo].[Uspload_MR_PR_Exception] '2020-01-01' ,'2020-06-30'

CREATE PROCEDURE  [dbo].[Uspload_MR_PR_Exception]
(
	@StartDate Date, @EndDate Date
)
AS
BEGIN

--#### Test Procedure ####
--Declare @StartDate Date, @EndDate Date
--Set @StartDate = '2020-01-01'
--Set @EndDate   = '2020-06-30'
----#### Test Ends ####

IF OBJECT_ID('tempdb..#MR_PR_Exceptions_Data') IS NOT NULL DROP TABLE #MR_PR_Exceptions_Data
SELECT 
      [LOANNUMBER]   								     = [Loan number new]								
     ,[Product_Code]									 = [Product Code]									
     ,[Product_Description]							     = [Product Description]							
     ,[Loan_Type]										 = [Loan Type]										
     ,[Exception_Flag]									 = [Exception Flag]									
     ,[Expanded_Authority_Y_N]							 = [Expanded Authority Y/N]							
     ,[Interest_rate]									 = [Interest rate]									
     ,[Total_points_Paid]								 = [Total points Paid]								
     ,[Manual_Margin_Adjustment]						 = [Manual Margin Adjustment]						
     ,[Manual_Points_Adjustment]						 = [Manual Points Adjustment]						
     ,[Manual_Rate_Adjustment]							 = [Manual Rate Adjustment]							
     ,[Total_Loan_Level_price_Adjustments]				 = [Total Loan Level price Adjustments]				
     ,[Total_Manual_Points_Adjustment_Sales]			 = [Total Manual Points Adjustment - Sales]			
     ,[Total_Manual_Points_Adjustment_Operations]		 = [Total Manual Points Adjustment Operations]		
     ,[Required_Price]									 = [Required Price]									
     ,[Loan_Amount]									     = [Loan Amount]									
     ,[Property_Type]									 = [Property Type]									
     ,[Occupancy]										 = [Occupancy]										
     ,[Property_State]									 = [Property State]									
     ,[Property_County]								     = [Property County]								
     ,[Sales_Site]										 = [Sales Site]										
     ,[Loan_Officer]									 = [Loan Officer]									
     ,[Application_Date]								 = [Application Date]								
     ,[File_Received_Date]								 = [File Received Date]								
     ,[Estimated_Closing_Date]							 = [Estimated Closing Date]							
     ,[Closing_Date]									 = [Closing Date]									
     ,[Loan_Status]									     = [Loan Status]																					 
     ,[Manual_Adjustment_Description]				     = [Manual Adjustment - Description]
	 ,Segment                                            = Case 
	                                                         when [Manual Adjustment - Description] Like '%Match Competitor Pricing%'	Then 'Competitive Match Pricing Exceptions'	
															 Else 'Other'
														   End
     ,[ACH_Y_N]										     = [ACH Y/N]										
     ,[Branch_Number]									 = [Branch #]										
     ,[Total_Loan_Level_Rate_Adjustments]				 = [Total Loan Level Rate Adjustments]				
     ,[Total_Loan_Level_Margin_Adjustments]			     = [Total Loan Level Margin Adjustments]			
     ,[Data_Period]									     = [Data_Period]	
	 ,[Data_Month]                                       = Eomonth(cast(left(ltrim(rtrim([Data_Period])),2) + '-01-' + right(ltrim(rtrim([Data_Period])),02) as Date))
Into #MR_PR_Exceptions_Data							
FROM Fair_Lending.[dbo].[MR_PR_Exceptions_Data]   With (NoLock)
Where (Eomonth(cast(left(ltrim(rtrim([Data_Period])),2) + '-01-' + right(ltrim(rtrim([Data_Period])),02) as Date)) between @StartDate and @EndDate)




IF OBJECT_ID('tempdb..#FL_Mortgage_Raw') IS NOT NULL DROP TABLE #FL_Mortgage_Raw
Select 
   *
   ,BookedStatus = Case when ACTIONTAKEN = 1 then 'Booked' ELse 'Not Booked' End
   ,[RowNum]     = ROW_NUMBER ( ) OVER (  PARTITION BY LOANNUMBER  order by ACTIONDATE Asc)  
Into #FL_Mortgage_Raw
From Fair_Lending.[dbo].[FL_Mortgage_Raw]            fl_raw with (Nolock)
Where ACTIONDATE between @StartDate and @EndDate 
		And PLANDD not in
		(
		 'CHFA Conventional'
		,'CHFA DAP 2ND'
		,'CHFA FHA'
		,'CT HOUSING CONV'
		,'CT HOUSING FHA'
		,'KEYSTONE HOME LOAN CONV'
		,'KEYSTONE HOME LOAN FHA'
		,'MHP'
		,'MHP 1ST MTG SOFT SEC'
		,'MHP 2nd MTG S/S >90%'
		,'NH HOUSING FHA'
		,'NO COST/LOW COST REFI MORTGAGE - Payment Shock'
		,'NO COST/LOW COST REFINANCE MORTGAGE'
		,'NO COST/LOW COST REFINANCE MORTGAGE EXCEPTION'
		,'PHFA 97% with no MI'
		,'PHFA HFA Preferred Risk Sharing'
		,'PHFA Keystone Advantage 2nd'
		,'PHFA Keystone Home Loan - Conventional'
		,'PORTFOLIO DHM PLUS'
		,'PORTFOLIO DHM PLUS EXCEPTION'
		,'RIH 1ST HOMES FHA'
		,'RIH 2ND MORTGAGE'
		,'SONYMA ACHIEVE THE DREAM'
		,'SONYMA Achieving the Dream'
		,'SONYMA LOW INTEREST FIXED RATE'
		,'SONYMA LOW INTEREST RATE'
		)




IF OBJECT_ID('tempdb..#BasePopulation') IS NOT NULL DROP TABLE #BasePopulation
SELECT
       exc.*
      ,mr.[ACTIONDATE]
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
      ,mr.BO_RELATION_CD
	  ,mr.OG_EMPL_CD
	  ,mr.[RowNum]
	  ,RACEETH_NAME                  = Case 
	                                     When APPETHINICITY = 1 then 'Hispanic'
										 When APPETHINICITY = 2 and APPRACE2 > 0  then 'Multiracial'
										 When APPETHINICITY = 2 and Ltrim(Rtrim(APPRACE1)) = '1'  then 'American Indian or Alaskan'
										 When APPETHINICITY = 2 and Ltrim(Rtrim(APPRACE1)) in ('2','4') then 'Asian or Pacific Islander'
										 When APPETHINICITY = 2 and Ltrim(Rtrim(APPRACE1)) = '3' then 'African American'
										 When APPETHINICITY = 2 and Ltrim(Rtrim(APPRACE1)) = '5' then 'Non-Hispanic White'
										 Else 'Not Available'
									   End
	  ,CORACEETH_NAME                = Case 
	                                     When (COAPPLASTNAME is  null or Len(Ltrim(Rtrim(COAPPLASTNAME)))	 = '0')  Then 'No Coapplicant'  
	                                     When COAPPETHINICITY = 1 then 'Hispanic'
										 When COAPPETHINICITY = 2 and cast(COAPPRACE2 as int) > 0  then 'Multiracial'
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
FROM #FL_Mortgage_Raw mr 
     Left join #MR_PR_Exceptions_Data exc on exc.LOANNUMBER = mr.LOANNUMBER and mr.RowNum = 1
Where mr.ACTIONTAKEN in (1) 
      and (Isnull(mr.OG_EMPL_CD,0) not in ('1','9','A','B','C','D','E','F','G','H','K','L','M','N','O','P','R','V','W') )


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
	  ,Is_Exception_Approved         = Case When Cast([Total_Manual_Points_Adjustment_Sales] as Float) < 0 Then 1 Else 0 End
	  ,Is_Exception_Declined         = Case When Cast([Total_Manual_Points_Adjustment_Sales] as Float) = 0 Then 1 Else 0 End
Into #FinalBasePopulation
From #BasePopulation


Truncate Table Fair_Lending.dbo.MR_PR_Exceptions_Data_Analysis

Insert into Fair_Lending.dbo.MR_PR_Exceptions_Data_Analysis
Select * from #FinalBasePopulation

End



GO


