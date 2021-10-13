"""
Created on Tue Feb 25 18:10:38 2020

@author: DADM Group
"""

"""Data Sources
#https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2020/pvqr-7yc4
#https://data.cityofnewyork.us/Transportation/DOF-Parking-Violation-Codes/ncbg-6agr   
https://data.ny.gov/Transportation/Vehicle-Makes-and-Body-Types-Most-Popular-in-New-Y/3pxy-wy2i
https://dmv.ny.gov/registration/registration-class-codes
https://www.purdue.edu/business/risk_mgmt/Other_Links/Vehicle_Body_Style_Codes.html
http://www.nyc.gov/html/dof/html/pdf/faq/stars_codes.pdf

https://nypost.com/2016/01/18/your-car-will-probably-never-get-towed-in-staten-island/

Stats ->
https://realpython.com/python-statistics/
"""

#measure the time that the script runs
import time
t0 = time.time()

#import libraries needed for data analysis
import math
import statistics
import scipy.stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

import pandas as pd
import numpy as np

#see if you can bring in 3 years worth of data

#import files into python
df1 = pd.read_csv(r'C:\Users\jayes\Downloads\Parking_Violations_Issued_-_Fiscal_Year_2020.csv')
df2 = pd.read_csv(r'C:\Users\jayes\Downloads\Parking_Violations_Issued_-_Fiscal_Year_2019.csv')
violationcode = pd.read_csv(r'C:\Users\jayes\Downloads\violationcodes.csv')
vehiclebodytype = pd.read_csv(r'C:\Users\jayes\Downloads\VehicleBodyStyles.csv')
vehicleplatetype = pd.read_csv(r'C:\Users\jayes\Downloads\vehicleplatetype.csv')
violationtime = pd.read_csv(r'C:\Users\jayes\Downloads\violationtime.csv')
colorcode = pd.read_csv(r'C:\Users\jayes\Downloads\colorcode.csv')
borough = pd.read_csv(r'C:\Users\jayes\Downloads\borough.csv')
vehicleclass = pd.read_csv(r'C:\Users\jayes\Downloads\vehicleclass.csv')

#clean columnn names to create a standardized format
df1.columns = df1.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(', '').str.replace(')', '')
df2.columns = df2.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(', '').str.replace(')', '')
violationcode.columns = violationcode.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(', '').str.replace(')', '')
vehiclebodytype.columns = vehiclebodytype.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(', '').str.replace(')', '')
violationtime.columns = violationtime.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(', '').str.replace(')', '')
vehicleplatetype.columns = vehicleplatetype.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(', '').str.replace(')', '')
colorcode.columns = colorcode.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(', '').str.replace(')', '')
vehicleclass.columns = vehicleclass.columns.str.strip().str.lower().str.replace(' ','_').str.replace('(', '').str.replace(')', '')


violationcode.rename(columns = {'code': 'violation_code'}, inplace = True) #rename column name to violation code from the pc dataset. Required for merge to work
vehiclebodytype.rename(columns = {'body_code': 'vehicle_body_type'}, inplace = True)
colorcode.rename(columns = {'color_code': 'vehicle_color'}, inplace = True)   

"""Start Append the two datasets"""
park = df1.append(df2) #Append 2020 & 2019 Parking data together 
del df1
del df2
"""End Append the two datasets"""

park.dropna()

"""Start Drop irrelevant columns"""
park = park.drop(columns = ['plate_id', 'issuing_agency','street_code1','street_code2','street_code3', \
                            'vehicle_expiration_date','violation_location','violation_precinct','issuer_precinct',\
                            'issuer_code','issuer_command','issuer_squad','time_first_observed',\
                            'violation_in_front_of_or_opposite','house_number','street_name','intersecting_street',\
                            'date_first_observed','law_section','sub_division','violation_legal_code','days_parking_in_effect',\
                            'from_hours_in_effect','to_hours_in_effect','unregistered_vehicle?','meter_number','feet_from_curb',\
                            'violation_post_code','violation_description','no_standing_or_stopping_violation','hydrant_violation',\
                            'double_parking_violation'])
"""End Drop irrelevant columns"""


"""Start to create new fields for Year, Month, Day"""
park['year'] = pd.DatetimeIndex(park['issue_date']).year
park['month_no'] = pd.DatetimeIndex(park['issue_date']).month
park['day'] = pd.DatetimeIndex(park['issue_date']).day
"""End to create new fields for Year, Month, Day"""

"""Start extraction for just year 2019"""
park = park.loc[park['year'] == 2019] #Subset only rows with 2019 data 
"""End extraction for just year 2019"""

"""Start Build Season Dictionary"""
#build list and convert to dataframe 
season = [{'1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8','9':'9', '10':'10', '11':'11', '12':'12'},\
          {'1':'Jan', '2':'Feb', '3':'Mar', '4':'Apr', '5':'May', '6':'Jun', '7':'Jul', '8':'Aug','9':'Sep', '10':'Oct', '11':'Nov', '12':'Dec'}, \
          {'1':'Winter', '2':'Winter', '3':'Spring', '4':'Spring', '5':'Spring', '6':'Summer', '7':'Summer', '8':'Summer','9':'Fall', '10':'Fall', '11':'Fall', '12':'Winter'}]
    
season = pd.DataFrame(season, index = ['month_no','month_name','season'])          #append column names
season = season.transpose()                                                             #transpose the dataset 
season['month_no'] = season['month_no'].astype(int)                                     #convert to integer/required for merge
"""End Build Season Dictionary"""


"""Start Merge datasets"""
park = pd.merge(park, violationcode, how='outer', on='violation_code')
park = pd.merge(park, season, how = 'outer', on='month_no' )                   #creates Month, Day, Season
park = pd.merge(park, colorcode, how = 'outer', on='vehicle_color')            #create color mapppings with color descriptions
park = pd.merge(park, vehicleplatetype, how = 'outer', on ='plate_type')
park = pd.merge(park, vehiclebodytype, how = 'outer', on ='vehicle_body_type')
park = pd.merge(park, violationtime, how = 'outer', on ='violation_time')
park = pd.merge(park, borough, how = 'outer', on ='violation_county')
park = pd.merge(park, vehicleclass, how = 'outer', on ='vehicle_make')

"""End Merge datasets"""


"""Drop all null values for any given row"""



"""Get aggregated counts per unique row value"""
#summons_number = park.summons_number.value_counts()
registration_state = park.registration_state.value_counts()
plate_type = park.plate_type.value_counts()
issue_date = park.issue_date.value_counts()
violation_code = park.violation_code.value_counts()
vehicle_body_type = park.vehicle_body_type.value_counts()
vehicle_make = park.vehicle_make.value_counts()
violation_definition = park.violation_definition.value_counts()
violation_time = park.violation_time.value_counts()
violation_county = park.violation_county.value_counts()
vehicle_color = park.vehicle_color.value_counts()
vehicle_year = park.vehicle_year.value_counts()
plate_category = park.plate_category.value_counts()
body_style_description = park.body_style_description.value_counts()
#vehicle_class = park['vehicleclass'].value_counts()


#fine
conditions = [
    (park['borough'] == 'Manhattan'),
    (park['borough'] != 'Manhattan')
    ]

choices = [
    (park['manhattan']),
    (park['other'])
    ]
park['fine'] = np.select(conditions, choices)


"""timer"""
t1 = time.time()
total = [(t1 - t0)/60]
print (total)
"""timer"""

#Regression Data Preparation
#filter to just Manhattan
dropborough = pd.DataFrame(['Manhattan'],columns = ['borough'])
park = park.loc[park.borough.isin(dropborough['borough'])]

#filter to just Passenger, Commercial
dropplatecategory = pd.DataFrame(['Passenger','Commercial','Livery, Taxi & Transport'],columns = ['plate_category'])
park = park.loc[park.plate_category.isin(dropplatecategory['plate_category'])]

#drop "Drop" from time 
droptime = pd.DataFrame(['Drop'],columns = ['time'])
park = park.loc[~park.mae.isin(droptime['time'])]

#drop specific colors
dropcolors  = pd.DataFrame(['Tan', 'Maroon', 'Orange', 'Gold', 'Purple', 'Pink', 'No Color', 'Dark','Brown','Yellow'],columns = ['color_description'])
park = park.loc[~park.color_description.isin(dropcolors['color_description'])]

#drop "Other" from bucketing
dropbucket = pd.DataFrame(['Other'],columns = ['bucket'])
park = park.loc[~park.bucketing.isin(dropbucket['bucket'])]

#filter costcat to just low
dropcostcat = pd.DataFrame(['Low'],columns = ['costcat'])
park = park.loc[park.costcat.isin(dropcostcat['costcat'])]


#filter to top 5 violations
dropviolation = pd.DataFrame(['PHTO SCHOOL ZN SPEED VIOLATION','NO PARKING-STREET CLEANING','FAIL TO DSPLY MUNI METER RECPT','NO STANDING-DAY/TIME LIMITS', 'NO PARKING-DAY/TIME LIMITS'],columns = ['violation'])
park = park.loc[park.violation_definition.isin(dropviolation['violation'])]


park.fine.value_counts()
park.bucketing.value_counts()
park.color_description.value_counts()
park.mae.value_counts()
park.costcat.value_counts()
park.plate_category.value_counts()

#subset columns
park = park[['season', 'color_description','plate_category', 'body_style_description', 'mae', 'ampm', 'borough','class', 'fine', 'costcat']]

#drop na values 
park = park.dropna()


#Regression Testing
"""
del dependant
del independant
del regressiondata
del model

regressiondata = pd.get_dummies(park9, columns=['mae'])
regressiondata = regressiondata.dropna()
regressiondata.columns

dependant = regressiondata['fine']
independant = regressiondata[['mae_Afternoon', 'mae_Evening', 'mae_Morning']]

#Add Constant
independant = sm.add_constant(independant)

#Run Regression
model = sm.OLS(dependant,independant).fit()
print(model.summary())
"""

x1 = pd.get_dummies(park, columns=['mae'])
x1 = x1.dropna()
x1.columns

b = x1['fine']
a = x1[['mae_Afternoon', 'mae_Evening', 'mae_Morning']]

x1 = x1.dropna()

#Add Constant
a = sm.add_constant(a)

#Run Regression
model = sm.OLS(b,a).fit()
print(model.summary())

#Plot Test#
import pandas as pd
import statsmodels.api as sm
import pylab
import math
import numpy as np
import matplotlib.pyplot as plt 
###%matplotlib inline

from scipy.stats import norm


import scipy.stats
import warnings
warnings.filterwarnings("ignore")
 
# =============================================================================
# df=pd.read_csv("violation_time_amount.csv",sep=",",names=['violation_time','fine_amount'],header=1)
# df.head()
# df.describe()
# =============================================================================


#hypothesis test prerequisite: normal distribute
observed_fine_amount = park['fine'].sort_values()
bin_val = np.arange(start = observed_fine_amount.min(),stop=observed_fine_amount.max(),step=10)
mu,std = np.mean(observed_fine_amount),np.std(observed_fine_amount)
 
p = norm.pdf(observed_fine_amount, mu,std)
 
plt.hist(observed_fine_amount,bins=bin_val,normed=True,stacked=True)
plt.plot(observed_fine_amount,p,color='r')
plt.xticks(np.arange(0,600,50),rotation=90)
plt.xlabel('Fine Amount Distributions')
plt.ylabel('Fine Amount')
plt.show()
print("Average (Mu):"+str(mu)+"/ Standard Deviation:" + str(std))


x = observed_fine_amount

#null hypothesis: x from normal distribution 
#As pvalue is 0, reject null hypothesis
k2,p = scipy.stats.normaltest(observed_fine_amount)
print("k2:",k2,"p:",p)

#determining normality is through probability–probability plot
scipy.stats.probplot(observed_fine_amount,dist='norm',plot=pylab)
pylab.show()


#null hypothesis: mean is equal to 72
#As pvalue is 0,  reject null hypothesis
CW_mu = 72
ttest_1samp=scipy.stats.ttest_1samp(park['fine'],CW_mu,axis=0)

print(ttest_1samp)