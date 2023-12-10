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
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier

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
df = pandas.read_csv('preprocessed_dataset/label_encoding/ML_MED_Dataset_test_preprocessed_label.csv')
X_test = df.iloc[:,1:17]
y_test = df.iloc[:,17:22]
y_test = np.ravel(y_test)
#base estimators best_params
svc_best_params = {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
sgd_best_params = {'alpha': 0.01, 'loss': 'modified_huber', 'penalty': 'l2'}
dtc_best_params = {'criterion': 'entropy', 'splitter': 'best'}
#training and hyperparameters tuning
gnb = GaussianNB()
svc = SVC()
sgd = SGDClassifier()
dtc = DecisionTreeClassifier()
distribution_params = {'estimator':[gnb,svc,sgd,dtc],'n_estimators':scipy.stats.poisson(loc=200, mu=75), 'algorithm':['SAMME'],'learning_rate':scipy.stats.uniform(loc=0, scale=2)}
adb = AdaBoostClassifier()
print("start \n")
adb_randomized = RandomizedSearchCV(estimator=adb, param_distributions=distribution_params, n_jobs=-1, n_iter=200, cv=8,verbose=3)
adb_randomized.fit(X_train, y_train)
print(adb_randomized.best_params_)
adb = AdaBoostClassifier(**adb_randomized.best_params_)
adb.fit(X_train,y_train)
y_pred = adb.predict(X_validation)
f_one_score = f1_score(y_validation, y_pred, average="weighted")
accuracy = accuracy_score(y_validation, y_pred)
print("f1 score:", f_one_score)
print("accuracy score:", accuracy)
