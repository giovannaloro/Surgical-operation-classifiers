from numpy import mean
from numpy import std
from numpy import ravel
import pandas 
import os 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
#uploading dataset
os.chdir('..')
df = pandas.read_csv('preprocessed_dataset//label_encoding/ML_MED_Dataset_validation_preprocessed_label.csv')
X_kf = df.iloc[:,1:17]
y_kf = df.iloc[:,17:22]
y_kf = ravel(y_kf)
#crossvalidation best_params = {'algorithm': 'SAMME', 'estimator': SGDClassifier(), 'learning_rate': 0.10253081850828027, 'n_estimators': 284}

best_params = {'algorithm': 'SAMME', 'estimator': SVC(), 'learning_rate': 0.10253081850828027, 'n_estimators': 284}
cv = KFold(n_splits=8, random_state=1, shuffle=True)
rf = AdaBoostClassifier(**best_params)
scores = cross_val_score(rf, X_kf, y_kf, scoring='accuracy', cv=cv, n_jobs=-1)
print("accuracy : \n",mean(scores))
scores = cross_val_score(rf, X_kf, y_kf, scoring='f1_micro', cv=cv, n_jobs=-1)
print("f1_micro : \n",mean(scores))
scores = cross_val_score(rf, X_kf, y_kf, scoring='f1_macro', cv=cv, n_jobs=-1, verbose=3)
print("f1_macro : \n",mean(scores))

