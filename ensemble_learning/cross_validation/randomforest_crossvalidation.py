from numpy import mean
from numpy import std
import pandas 
import os 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
#uploading dataset
os.chdir('..')
df = pandas.read_csv('preprocessed_dataset/ML_MED_KFold_preprocessed.csv')
X_kf = df.iloc[:,1:17]
y_kf = df.iloc[:,17:22]
#crossvalidation
best_params={'criterion': 'entropy', 'max_features': 'log2', 'min_samples_split': 27, 'n_estimators': 322, 'n_jobs': -1}
cv = KFold(n_splits=8, random_state=1, shuffle=True)
rf = RandomForestClassifier(**best_params)
scores = cross_val_score(rf, X_kf, y_kf, scoring='accuracy', cv=cv, n_jobs=-1)
print("accuracy : \n",mean(scores))
scores = cross_val_score(rf, X_kf, y_kf, scoring='f1_micro', cv=cv, n_jobs=-1)
print("f1_micro : \n",mean(scores))
scores = cross_val_score(rf, X_kf, y_kf, scoring='f1_macro', cv=cv, n_jobs=-1)
print("f1_macro : \n",mean(scores))


