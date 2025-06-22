import pandas as pd
import numpy as np
import random
from sklearn import preprocessing
import kaggle

import os
from kaggle.api.kaggle_api_extended import KaggleApi
import json
from datetime import datetime, timedelta

def data_prep(data):
    """This function lets us do some generic data preprocessing to move forward"""
  
    #Converting all the data columns into the correct datatypes, correct formats, also added and age category. 
    data.drop(['Unnamed: 0','cc_num','first','last','street','zip','trans_num','lat','long','merchant','merch_lat','merch_long','gender','unix_time','category','city'],inplace=True,axis=1)
    
    #Converting the data columns into the correct datatype. 
    data['trans_date_trans_time']= pd.to_datetime(data['trans_date_trans_time'])
    data['month'] = data['trans_date_trans_time'].dt.month
    data['month'] = data['month'].astype(int)
    data['dob']= pd.to_datetime(data['dob'])
    data["job"]= data["job"].astype(str)
    
    #Lets find the age when the fraud happened and drop the dob column. 
    data['age'] = (data['trans_date_trans_time'] - data['dob']).dt.days/365
    data.drop(['trans_date_trans_time','dob'],inplace=True,axis=1)
    
    #Lets hot encode job and state, as XGBoost takes numerical values
    label_encoder = preprocessing.LabelEncoder()
    data["job"]= label_encoder.fit_transform(data["job"])
    data["state"]= label_encoder.fit_transform(data["state"])

    return data

kaggle.api.authenticate()
kaggle.api.dataset_download_files('kartik2112/fraud-detection', path='data/', unzip=True)

train = pd.read_csv("data/fraudTrain.csv")
test = pd.read_csv("data/fraudTest.csv")

train = data_prep(train)
test = data_prep(test)

#Lets find the class distribution 
print("Class Distribution:")
print(train['is_fraud'].value_counts())
print(f"Percentage: \n{train['is_fraud'].value_counts(normalize=True) * 100}")

#Lets get the imbalance ration
imbalance_ratio = train['is_fraud'].value_counts().min() / train['is_fraud'].value_counts().max()
print(f"Imbalance ratio: {imbalance_ratio:.3f}")

# Split instances into majority vs minority class/classes
df_majority = train[train['is_fraud'] == 0]
df_minority = train[train['is_fraud'] == 1]

# Undersampling: we keep as many majority instances (n) as minority ones
df_majority_downsampled = df_majority.sample(n=len(df_minority),random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])

df_balanced.to_csv('data/preprocessed_fraudTrain.csv')
test.to_csv('data/preprocessed_fraudTest.csv')