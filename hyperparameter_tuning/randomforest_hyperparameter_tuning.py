import pandas 
import os 
import scipy
import numpy as np 
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
#uplouading dataset 
os.chdir('..')
df = pandas.read_csv('preprocessed_dataset/ML_MED_Dataset_train_preprocessed.csv')
X_train = df.iloc[:,1:17]
y_train = df.iloc[:,17:22]
df = pandas.read_csv('preprocessed_dataset/ML_MED_KFold_preprocessed.csv')
X_kf = df.iloc[:,1:17]
y_kf = df.iloc[:,17:22]
df = pandas.read_csv('preprocessed_dataset/ML_MED_Dataset_validation_preprocessed.csv')
X_validation = df.iloc[:,1:17]
y_validation = df.iloc[:,17:22]
df = pandas.read_csv('preprocessed_dataset/ML_MED_Dataset_test_preprocessed.csv')
X_test = df.iloc[:,1:17]
y_test = df.iloc[:,17:22]
#training and hyperparamater tuning _____
rf = RandomForestClassifier()
distribution_params = {'n_estimators':scipy.stats.poisson(loc=250, mu=75), 'criterion':['gini','entropy','log_loss'],'max_features':['sqrt','log2'],
                       'min_samples_split':scipy.stats.poisson(loc=10, mu=25),'n_jobs':[-1]}
rf_randomized = RandomizedSearchCV(estimator=rf, param_distributions=distribution_params, n_jobs=-1, n_iter=200, cv=8, verbose=3)
rf_randomized.fit(X_train,y_train)
print(rf_randomized.best_params_)
print(rf_randomized.best_score_)
#evaluation
rf = RandomForestClassifier(**rf_randomized.best_params_)
rf.fit(X_train, y_train)
y_pred =rf.predict(X_validation)
f_one_score = f1_score(y_validation, y_pred, average="weighted")
accuracy = accuracy_score(y_validation, y_pred)
print("f1 score:", f_one_score)
print("accuracy score:", accuracy)