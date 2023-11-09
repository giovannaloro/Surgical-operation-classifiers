import pandas
import os 
import scipy 
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
#uplouading dataset 
os.chdir('..')
df = pandas.read_csv('preprocessed_dataset/label_encoding/ML_MED_Dataset_train_preprocessed_label.csv')
X_train = df.iloc[:,1:17]
y_train = df.iloc[:,17:22]
y_train = np.ravel(y_train)
df = pandas.read_csv('preprocessed_dataset/label_encoding/ML_MED_KFold_preprocessed_label.csv')
X_kf = df.iloc[:,1:17]
y_kf = df.iloc[:,17:22]
y_kf = np.ravel(y_kf)
df = pandas.read_csv('preprocessed_dataset/label_encoding/ML_MED_Dataset_validation_preprocessed_label.csv')
X_validation = df.iloc[:,1:17]
y_validation = df.iloc[:,17:22]
y_validation = np.ravel(y_validation)
#training and hyperparameters tuning
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_validation)
f_one_score = f1_score(y_validation, y_pred, average="weighted")
accuracy = accuracy_score(y_validation, y_pred)
print("f1 score:", f_one_score)
print("accuracy score:", accuracy)
