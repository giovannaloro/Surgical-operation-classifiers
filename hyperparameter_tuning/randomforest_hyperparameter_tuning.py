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
df = pandas.read_csv('preprocessed_dataset/ML_MED_Dataset_train_preprocessed.csv')
X_train = df.iloc[:,1:17]
y_train = df.iloc[:,17:22]
df = pandas.read_csv('preprocessed_dataset/ML_MED_KFold_preprocessed.csv')
X_kf = df.iloc[:,1:17]
y_kf = df.iloc[:,17:22]
df = pandas.read_csv('preprocessed_dataset/ML_MED_Dataset_validation_preprocessed.csv')
X_validation = df.iloc[:,1:17]
y_validation = df.iloc[:,17:22]
#training and hyperparamater tuning _____________
rf = RandomForestClassifier()
distribution_params = {'n_estimators':scipy.stats.poisson(loc=500, mu=75), 'criterion':['gini','entropy','log_loss'],'min_samples_split':scipy.stats.poisson(loc=10, mu=25),'max_features':['sqrt','log2']}
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

