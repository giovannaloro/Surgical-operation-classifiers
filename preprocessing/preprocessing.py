import pandas as pd
import os 
import scipy 
import numpy as np 
from joblib import dump, load
from sklearn import datasets
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#uplouading dataset 
datasets=['MLMED_Dataset.csv','ML_MED_Dataset_validation.csv','ML_MED_Dataset_test.csv','ML_MED_KFold.csv','ML_MED_Dataset_train.csv']
for dataset in datasets:
    df = pd.read_csv(f'dataset/{dataset}')
    X_num = df.iloc[:,0:4]
    X_cat = df.loc[:,['ENDOSCOPIA','LAPAROSCOPIA','OPEN','TORACOSCOPIA','Antiaggreganti','ASA_3.0','ASA_2.0','Antipertensivi','Ipertensione arteriosa','Altro_terapia','Fumo','Altro_comorbidita']]
    #scaling and selecting dataset
    scaler_num_train = preprocessing.StandardScaler().fit(X_num)
    X_num = scaler_num_train.transform(X_num)
    X_num = pd.DataFrame(X_num, columns = ['Età','Peso','Altezza','BMI'])
    X = X_num.join(X_cat)
    y = df.iloc[:,46:51]
    output_dataset = X.join(y)
    #saving preprocessed dataset
    file_name = dataset.split('.')[0] + '_preprocessed' + '.csv'
    output_dataset.to_csv(file_name)
