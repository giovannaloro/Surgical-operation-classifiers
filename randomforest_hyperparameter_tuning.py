import pandas 
import os 
import scipy 
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score
#uplouading dataset 
df = pandas.read_csv('dataset/ML_MED_Dataset_train.csv')
df = pandas.read_csv('dataset/ML_MED_KFold.csv')
X_train = df.iloc[:,0:46]
y_train = df.iloc[:,46:51]
X_kf = df.iloc[:,0:46]
y_kf = df.iloc[:,46:51]
df = pandas.read_csv('dataset/ML_MED_Dataset_validation.csv')
X_validation = df.iloc[:,0:46]
y_validation = df.iloc[:,46:51]
#normalizing set 
scaler_x_kf = preprocessing.StandardScaler().fit(X_kf)
scaler_x_train = preprocessing.StandardScaler().fit(X_train)
scaler_x_validation = preprocessing.StandardScaler().fit(X_validation)
X_kf = scaler_x_kf.transform(X_kf)
X_validation = scaler_x_validation.transform(X_validation)
X_train = scaler_x_train.transform(X_train)
#training and hyperparamater tuning _____________
rf = RandomForestClassifier()
distribution_params = {'n_estimators':scipy.stats.poisson(loc=200, mu=75), 'criterion':['gini','entropy','log_loss'],'min_samples_split':scipy.stats.poisson(loc=10, mu=25),'max_features':['sqrt','log2']}
rf_randomized = RandomizedSearchCV(estimator=rf, param_distributions=distribution_params, n_jobs=-1, n_iter=20, cv=8)
rf_randomized.fit(X_kf,y_kf)
best_rf = rf_randomized.best_estimator_
print(rf_randomized.best_params_)
#evaluation
best_rf.fit(X_train, y_train)
y_pred =best_rf.predict(X_validation)
f_one_score = f1_score(y_validation, y_pred, average="weighted")
accuracy = accuracy_score(y_validation, y_pred)
print("f1 score:", f_one_score)
print("accuracy score:", accuracy)

