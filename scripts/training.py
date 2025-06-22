import pandas as pd
import numpy as np
import random
from sklearn import preprocessing

import xgboost as xgb
import sklearn
from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import GridSearchCV, train_test_split
import optuna
from datetime import datetime, timedelta

from tuning import find_best_parameters
import mlflow

uri = mlflow.get_tracking_uri()
mlflow.set_tracking_uri(uri)
mlflow.set_experiment("Fraud-detection")
mlflow.xgboost.autolog()

df_balanced = pd.read_csv('data/preprocessed_fraudTrain.csv')
test = pd.read_csv('data/preprocessed_fraudTest.csv')

X = df_balanced.drop(columns=['is_fraud'], axis=1)
y = df_balanced['is_fraud'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)

dmatrix_train = xgb.DMatrix(data=X_train, label=y_train)
dmatrix_test = xgb.DMatrix(data=X_test,label=y_test)

best_parameters = find_best_parameters(dmatrix_train,dmatrix_test, y_test)
print(best_parameters)