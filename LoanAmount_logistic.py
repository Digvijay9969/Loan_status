
#loading library
import os
import seaborn as sns 
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import figure


#setting directory path
os.getcwd()
os.chdir("D:/D_S/loan_Status_datascience_project")

#loading files
rawDf=pd.read_csv("train.csv")
predDf=pd.read_csv("test.csv")

rawDf.shape
rawDf.columns

predDf.shape
predDf.columns

#Adding Loan_Status column in test dataset testDf

predDf["Loan_Status"]="No"
predDf.columns

# sampling rawDf into train data and test data
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawDf,train_size=0.8,random_state=9969)

trainDf.shape
testDf.shape

#Adding source column in trainDf, testDf, predDf

trainDf["Source"]="Train"
testDf["Source"]="Test"
predDf["Source"]="Prediction"

#combine train, test and prediction datasets
fullRaw=pd.concat([trainDf,testDf,predDf],axis=0)
fullRaw.columns

#% split of 0s and 1s
fullRaw.loc[fullRaw['Source']== "Train",'Loan_Status'].value_counts()/fullRaw[fullRaw["Source"]=='Train'].shape[0]
#68% yes and 32% no

#summarize the data
fullRaw_summary = fullRaw.describe()

#Removing identifire column
fullRaw.drop(["Loan_ID"],axis=1,inplace=True)
fullRaw.columns
#missing value check
fullRaw.isna().sum()

#we have missing values in Gender, Married,Dependents,self_employed,loanamount,loanamount term, credit history

#Automated code for nullvalues fillied with meadian or mode

for col in fullRaw.columns : 
    
    if((fullRaw[col].isna().sum())>0): 
        print("Null values found in column :*",col)
        if((fullRaw[col].dtypes == "float64") or (fullRaw[col].dtypes== "int64")):
          tempMedian=fullRaw.loc[fullRaw["Source"]=="Train",col].median()
          tempMedian
          fullRaw[col].fillna(tempMedian,inplace=True)
        elif (fullRaw[col].dtypes == "O"):
            tempMode=fullRaw.loc[fullRaw["Source"]=="Train",col].mode()[0]
            tempMode
            fullRaw[col].fillna(tempMode,inplace=True)
    else:
        print("\tNO NUll values found in :",col)

######################33
#use data description excel sheet to convert numeric variables to categorical variables
# categorical variables : 

variableToUpdate = 'Credit_History'
#to check the unique categories of the variable
fullRaw[variableToUpdate].value_counts()
fullRaw[variableToUpdate].replace({0:"Bad",1:"Good"},inplace=True)
fullRaw[variableToUpdate].value_counts()


###############################
# Bivariate analysis using box plot
##############################
trainDf=fullRaw.loc[fullRaw["Source"]=="Train"]
continuousVars=trainDf.columns[trainDf.dtypes != object]
continuousVars

fileName="Continuous_Bivariate_Analysis.pdf"
pdf=PdfPages(fileName)
for colNumber,colName in enumerate(continuousVars): #enumerate gives key, value pair
    #print(colNumber,columns)
    figure()
    sns.boxplot(y=trainDf[colName],x=trainDf["Loan_Status"])
    pdf.savefig(colNumber+1) # colNumber+1 is done to ensure page numbering starts from 1
    
pdf.close()

#################33
# Bivariate analysis using Histogram
###################3
categoricalVars = trainDf.columns[trainDf.dtypes == object]
categoricalVars

sns.histplot(trainDf,x="Gender",hue="Loan_Status",stat="probability",multiple="fill")

fileName= "Categorical_Bivariate_Analysis_hist.pdf"
pdf=PdfPages(fileName)
for colNumber,colName in enumerate(categoricalVars):
    figure()
    sns.histplot(trainDf,x=colName,hue="Loan_Status",stat="probability",multiple="fill")
    pdf.savefig(colNumber+1)# colNumber+1 is done to ensure page numbering starts from 1 (and no 0)
pdf.close()


#####################
#change Dependent variable category manually
########################
fullRaw["Loan_Status"]=np.where(fullRaw["Loan_Status"]=="N",1,0)

##################################
#Dummy variable creation
#################################

fullRaw2=pd.get_dummies(fullRaw,drop_first = True)# 'source' column will change to 'source_Train' and it contains 0s and 1s
fullRaw2.shape

########################3
#Divide the data into Train and Test
############################3

#Divide the data into Train and Test based on source and column and make sure you drop the source column

#step 1:Divide into Train and Test
Train= fullRaw2[fullRaw2['Source_Train']==1].drop(['Source_Train','Source_Test'],axis=1).copy()
Train.shape

Test=fullRaw2[fullRaw2['Source_Test']==1].drop(['Source_Test','Source_Train'],axis=1).copy()
Test.shape

Prediction=fullRaw2[(fullRaw2['Source_Train']==0) & (fullRaw2['Source_Test']==0)].drop(["Source_Train","Source_Test"],axis=1)
Prediction.shape

#step 2: Divide  independent and dependent columns

depVar = "Loan_Status"
trainX = Train.drop([depVar], axis = 1).copy()
trainY = Train[depVar].copy()

testX = Test.drop([depVar], axis = 1).copy()
testY = Test[depVar].copy()

predictionX=Prediction.drop([depVar],axis=1).copy()

#################################333
# Add intercept column
#####################################
from statsmodels.api import add_constant
trainX=add_constant(trainX)
testX=add_constant(testX)
predictionX=add_constant(predictionX)
#################################
#VIF check
################################
from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF= 10 #This vif variable will be calculated at every iteration in the while loop
maxVIF=10
trainXcopy=trainX.copy()
counter=1
highVIFColumnNames=[]

while(tempMaxVIF >= maxVIF):
    print(counter)
    
    #create an empty temporary df to store VIF values
    tempVIFDf=pd.DataFrame()
    
    #calculate vif using list comprehsion
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXcopy.values, i)for i in range(trainXcopy.shape[1])]
    # creating newcolumn 'column_name' to store the col names against the vif values form list compherension
    tempVIFDf["column_name"]=trainXcopy.columns
    #Drop Na rows from the df if ther is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    #Sort the df based on vif values, then pick the top most column name (which has the highest vif)
    tempColumnName=tempVIFDf.sort_values(["VIF"],ascending=False).iloc[0,1]
    
    #store the max vif value in tempMaxVIF
    tempMaxVIF=tempVIFDf.sort_values(["VIF"],ascending=False).iloc[0,0]
    
    if(tempMaxVIF >= maxVIF): # This condition will ensure that columns having VIF lower than 10 are NOT dropped
    # Remove the highest VIF valued "Column" from trainXCopy. As the loop continues this step will keep removing highest VIF columns one by one 
      trainXcopy=trainXcopy.drop(tempColumnName,axis=1)
      highVIFColumnNames.append(tempColumnName)
      print(tempColumnName)
      
    counter= counter+1

highVIFColumnNames

highVIFColumnNames.remove('const') 
# we need to exclude const column from getting dropped or removed. This is intercept.
highVIFColumnNames

trainX=trainX.drop(highVIFColumnNames,axis=1)
testX=testX.drop(highVIFColumnNames,axis=1)
trainX.shape
testX.shape
predictionX = predictionX.drop(highVIFColumnNames, axis = 1)
predictionX.shape

#################################
#Model building
################################

#Build logistic regression model(using statsmodels packeges/library)
from statsmodels.api import Logit
M1 = Logit(trainY,trainX) #(dep_var,Indep_vars) #This is model defination
M1_Model=M1.fit()#This is model building
M1_Model.summary()#This is model output/summary

##########
#Manual model selection. Drop the most insignificant variable in model one by one
##############################3


#Drop Dependents_3+
colsToDrop=['Dependents_3+']
M2=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M2.summary()

#Drop Self_Employed_Yes  
colsToDrop.append('Self_Employed_Yes')
M3=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M3.summary()

#Drop Gender_Male 
colsToDrop.append('Gender_Male')
M4=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M4.summary()

#Drop Dependents_2
colsToDrop.append('Dependents_2')
M5=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M5.summary()

#drop Education_Not Graduate
colsToDrop.append('Education_Not Graduate')
M6=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M6.summary()


#Drop Property_Area_Urban  
colsToDrop.append('Property_Area_Urban')
M7=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M7.summary()


#Drop CoapplicantIncome
colsToDrop.append('CoapplicantIncome')
M8=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M8.summary()

#Drop Dependents_1
colsToDrop.append('Dependents_1')
M9=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M9.summary()

#Drop ApplicantIncome  
colsToDrop.append("ApplicantIncome")
M10=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M10.summary()

#Drop LoanAmount  
colsToDrop.append("LoanAmount")
M11=Logit(trainY,trainX.drop(colsToDrop,axis=1)).fit()
M11.summary()

##################################
# Prediction and validation
##################################

trainX=trainX.drop(colsToDrop,axis=1)
testX=testX.drop(colsToDrop,axis=1)
predictionX=predictionX.drop(colsToDrop,axis=1)

#storing probability prediction in testX
testX['Test_Prob']=M11.predict(testX) 
#a new column called Test_Prob should be created
testX.columns
testX['Test_Prob'][0:6]
testY[:6]

#Classify 0 or 1 based on 0.5 cutoff
testX['Test_class'] = np.where(testX['Test_Prob']>= 0.5,1,0)
testX.columns 

##############################
# Confusion Matrix
##############################

Confusion_Mat=pd.crosstab(testX['Test_class'],testY) #R,C format
Confusion_Mat

#Check accuracy of the model
(sum(np.diagonal(Confusion_Mat))/testX.shape[0])*100 # ~80%

#####################################
# Precision,Recall,F1 score
###################################

from sklearn.metrics import classification_report
print(classification_report(testY,testX['Test_class'])) #Actual, Predicted

#Precision: TP/(TP+FP)
#Recall : TP/(TP+FN)
#F1Score : 2*Precision*Recall/(Presion+Recall)
#Precision,Recall, F1 Score interpretation: Higher the better
#Precision
#Intuitive understanding : How many of our "Predicted" Loan status is "actual" Loan status
 
##################################
# AUC and ROC curve
##################################
from sklearn.metrics import roc_curve, auc
#predict on train data
Train_Prob = M11.predict(trainX)

#calculate FPR, TPR, Cutoff Thresholds
fpr,tpr,cutoff=roc_curve(trainY, Train_Prob)

# Cutoff Table Creation
Cutoff_Table= pd.DataFrame()
Cutoff_Table['FPR']=fpr
Cutoff_Table['TPR']=tpr
Cutoff_Table['Cutoff']=cutoff

#Plot Roc Curve
import seaborn as sns
sns.lineplot(Cutoff_Table['FPR'],Cutoff_Table['TPR'])

#Area under curve(AUC)
auc(fpr,tpr)

################################
# Improve model output using new cutoff point
#################################

Cutoff_Table['DiffBetweenTPRFPR']=Cutoff_Table['TPR']-Cutoff_Table['FPR']  #max diff between tpr and fpr
Cutoff_Table['Distance']=np.sqrt((1-Cutoff_Table['TPR'])**2 + (0-Cutoff_Table['FPR'])**2)
#New cutoff pont performance (obtained after stufying TOC Curve and cutoff)

cutoffPoint=0.8980285624424246









cutoffPoint

#classify the test prediction into classes of 0s and 1s
testX['Test_Class2']=np.where(testX['Test_Prob']>=cutoffPoint,1,0)
#Confussion Matrix
Confusion_Mat2=pd.crosstab(testX['Test_Class2'], testY)# R,C format
Confusion_Mat2

#Model Evaluation Metrics
print(classification_report(testY,testX['Test_Class2']))

#prediction on prediction dataset
predictionX["Loan_Status"]=M11.predict(predictionX)

predictionX["Loan_Status"]=np.where(predictionX["Loan_Status"]>=cutoffPoint,"N","Y")
output=pd.DataFrame()
output=pd.concat([predDf["Loan_ID"],predictionX["Loan_Status"]],axis=1)
output.to_csv("output.csv")

