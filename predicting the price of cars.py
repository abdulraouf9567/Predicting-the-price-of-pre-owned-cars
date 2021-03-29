# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:37:34 2021

@author: Raouf
"""

import numpy as np
import pandas as pd
import seaborn as sns

#setting the dimension for the plot
#===================================
sns.set(rc={'figure.figsize':(11.7,8.27)})
#====================================

#Reading the csv files
#====================================
cars_data =pd.read_csv('cars_sampled.csv')
#====================================

#copying the data
#====================================
cars=cars_data.copy()
#====================================

#====================================
#structure of the dataset
cars.info()
#====================================

#====================================
#summarizing the data
cars.describe()
pd.set_option('display.float_format',lambda x:'%.3f' % x)
cars.describe()

#====================================
#to display maximum set of columns
pd.set_option('display.max_columns',500)
#====================================

#====================================
#Dropping unwanted columns
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(col,axis=1)

#====================================
#Removing the duplicate data
cars.drop_duplicates(keep='first',inplace=True)

#====================================
#missing values
cars.isnull().sum()
#====================================
#variable yearofregistration
yearwise_count= cars['yearOfRegistration'].value_counts().sort_index()
sum(cars['yearOfRegistration']>2018)
sum(cars['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',data=cars,fit_reg=False,scatter=True)

#working range between 1950 and 2018
#variable price
price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxenplot(y=cars['price'])
sum(cars['price']<100)
sum(cars['price']>15000)
#working range 100 and 150000

#variable power ps
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'],kde=False)
cars['powerPS'].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',data=cars,scatter=True,fit_reg=False)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

#working range of data
cars=cars[
            (cars.yearOfRegistration <=2018)
          & (cars.yearOfRegistration>=1950)
          & (cars.price>=100)
          & (cars.price<=15000)
          & (cars.powerPS>=10)
          & (cars.powerPS<=500)]

# further to simplify => variable reduction
# combinig yearOfregistration and monthofregistration

cars['monthOfRegistration']/=12

cars['Age']=(2018-cars['yearOfRegistration']+cars['monthOfRegistration'])

cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

#dropping the yearofreg and month of reg
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#visualizing parameter
# Age 
sns.distplot(cars['Age'],kde=False)
sns.boxplot(y=cars['Age'])

#price
sns.distplot(cars['price'],kde=False)
sns.boxplot(y=cars['price'])

#powerps
sns.distplot(cars['powerPS'],kde=False)
sns.boxplot(y=cars['powerPS'])

#visualizing paramater after narrowing working range
#AGE VS PRICE
sns.regplot(x=cars['Age'],y=cars['price'],scatter=True,fit_reg=False)

#pwoer ps vs price
sns.regplot(x=cars['powerPS'],y=cars['price'],data=cars,scatter=True,fit_reg=False)

#variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)

#variable offer type
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)

#varible abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
sns.boxplot(x='abtest',y='price',data=cars)

#variable vehicle type
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)

#Variable GearBox

cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)

#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)

#variable kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x='kilometer',y='price',data=cars)

#varibale fueltype
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)

#varible brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)

#variable notRepairedDamage

cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)


#====================================
#Removing insignificant variables
col=['seller','abtest','offerType']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

#=========================
#correlation
cars_select1=cars.select_dtypes(exclude=['object'])
correlation=cars_select1.corr()
round(correlation,3)

"""We are going to build a linear regression and Random forest model
on two sets of data .
1.Data obtained by omitting rows with any missing value
2. Data obtained imputing the missing values"""


#=============================================
#omitting the missing values
#=============================================

cars_omit=cars.dropna(axis=0)
cars.isnull().sum()
cars_omit.isnull().sum()

#Converting categorical Data to dummy variables

cars_omit=pd.get_dummies(cars_omit,drop_first=True)
 

#imoporting necessary libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


#================================================
#Model Building with omitted DATA
#===============================================
x1=cars_omit.drop(['price'],axis='columns',inplace=False)
y1=cars_omit['price']


#Plotting the variable price

prices=pd.DataFrame({'1.Before':y1,'2.After':np.log(y1)})
prices.hist()

#Transforming prices as log values
y1=np.log(y1)

#splitting the train test values
X_train,X_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)

#====================================
#BASELINE MODEL FOR OMITTED DATA
#====================================

"""We are making a base model by using test data mean value
this is to set a benchmark and to compute with our regression model  """

#Finding the mean value
base_pred=np.mean(y_test)
print(base_pred)

#Repeating the same value till length of test data 
base_pred=np.repeat(base_pred,len(y_test))

#Findind RMSE
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)

#===================================
#LINEAR REGRESSION WITH OMITTED DATA
#===================================

lgr=LinearRegression(fit_intercept=True)

#MODEL FITTING
model_lin1=lgr.fit(X_train,y_train)

#predicting the model on test set 
cars_prediction_lin1 = lgr.predict(X_test)

#computing MSE and RMSE
lin_mse1=mean_squared_error(y_test, cars_prediction_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#R squared value
r2_lin_test1=model_lin1.score(X_test,y_test)
r2_lin_train1=model_lin1.score(X_train,y_train)
print(r2_lin_test1,r2_lin_train1)

#Regression Diagnostic - Residulal plot analysis
residuals1=y_test-cars_prediction_lin1
sns.regplot(x=cars_prediction_lin1,y=residuals1,scatter=True,fit_reg=False)

residuals1.describe()

#======================================================
#RandomForest with OMITTED DATA
#======================================================
rf=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_leaf=4,random_state=1,min_samples_split=10)

#model
model_rf1=rf.fit(X_train,y_train)

#predicting the model
cars_prediction_rf1=rf.predict(X_test)

#computing MSE and RMSE
rf_mse1=mean_squared_error(y_test,cars_prediction_rf1)
rf_rmse1=np.sqrt(rf_mse1)

#R squared value
r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1=model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)

#=====================================================
#model building with imputed data
#=====================================================
cars_imputed=cars.apply(lambda x:x.fillna(x.median()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

#converting categorical to dummy variables
cars_imputed = pd.get_dummies(cars_imputed,drop_first=True)

#=======================================================
#model building with imputed data
#=======================================================
#Seperating input and output

x2= cars_imputed.drop(['price'],axis='columns',inplace=False)
y2= cars_imputed['price']

#plotting the variable price
prices=pd.DataFrame({'1.Before':y2,'2.After':np.log(y2)})
prices.hist()

#Transforming prices as log values
y2=np.log(y2)

#splitting the train test values
X_train1,X_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)

#====================================
#BASELINE MODEL FOR OMITTED DATA
#====================================

"""We are making a base model by using test data mean value
this is to set a benchmark and to compute with our regression model  """

#Finding the mean value
base_pred=np.mean(y_test1)
print(base_pred)

#Repeating the same value till length of test data 
base_pred=np.repeat(base_pred,len(y_test1))

#Findind RMSE
base_root_mean_square_error_imputed=np.sqrt(mean_squared_error(y_test1,base_pred))
print(base_root_mean_square_error_imputed)

#===================================
#LINEAR REGRESSION WITH OMITTED DATA
#===================================

lgr2=LinearRegression(fit_intercept=True)

#MODEL FITTING
model_lin2=lgr.fit(X_train1,y_train1)

#predicting the model on test set 
cars_prediction_lin2 = lgr.predict(X_test1)

#computing MSE and RMSE
lin_mse2=mean_squared_error(y_test1, cars_prediction_lin2)
lin_rmse2=np.sqrt(lin_mse2)
print(lin_rmse2)

#R squared value
r2_lin_test2=model_lin1.score(X_test1,y_test1)
r2_lin_train2=model_lin1.score(X_train1,y_train1)
print(r2_lin_test2,r2_lin_train2)

#======================================================
#RandomForest with OMITTED DATA
#======================================================

rf2=RandomForestRegressor(n_estimators=100,max_features='auto',max_depth=100,min_samples_leaf=4,random_state=1,min_samples_split=10)

#model
model_rf2=rf2.fit(X_train1,y_train1)

#predicting the model
cars_prediction_rf2=rf2.predict(X_test1)

#computing MSE and RMSE
rf_mse2=mean_squared_error(y_test1,cars_prediction_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)

#R squared value
r2_rf_test2=model_rf2.score(X_test1,y_test1)
r2_rf_train2=model_rf2.score(X_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)

# =============================================================================
#========================================================
#Final Output
print("Metrics for models built from data where missing values were omitted")
print("R squared value for train from LR = ",r2_lin_train1)
print("R squared value for test from LR = ",r2_lin_test1)
print("R squared value for train from RF = ",r2_rf_train1)
print("R squared value for test from RF = ",r2_rf_test1)
print("BASE RMSE of model built from data where missing values were omitted = ",base_root_mean_square_error)
print("RMSE value for test from LR = ",lin_rmse1)
print("RMSE value for test from RF = ",rf_rmse1)
print("\n\n")
print("Metrics for models built from data where missing values were imputed")
print("R squared value for train from LR = ",r2_lin_train2)
print("R squared value for test from LR = ",r2_lin_test2)
print("R squared value for train from RF = ",r2_rf_train2)
print("R squared value for test from RF = ",r2_rf_test2)
print("BASE RMSE of model built from data where missing values were imputed = ",base_root_mean_square_error_imputed)
print("RMSE value for test from LR = ",lin_rmse2)
print("RMSE value for test from RF = ",rf_rmse1)



      
      
      
      
      
      
      
      
      
      
      
      
      
      )




