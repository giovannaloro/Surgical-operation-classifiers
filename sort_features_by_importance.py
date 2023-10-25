import pandas as pd
import numpy as np 
from joblib import dump, load
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
#feature selection with importances attribute of random forest
rf = load('models/best_randomforest_model.joblib') 
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = df.columns[:]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]],importances[indices[f]]))