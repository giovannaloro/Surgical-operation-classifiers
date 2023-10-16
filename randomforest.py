import pandas 
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score
#uplouading dataset 
df = pandas.read_csv("ML_MED_Dataset_train.csv")
X_train = df.iloc[:,0:46]
y_train = df.iloc[:,46:51]
df = pandas.read_csv("ML_MED_Dataset_validation.csv")
X_validation = df.iloc[:,0:46]
y_validation = df.iloc[:,46:51]
#normalizing set 
scaler_x_train = preprocessing.StandardScaler().fit(X_train)
scaler_x_validation = preprocessing.StandardScaler().fit(X_validation)
X_train = scaler_x_train.transform(X_train)
X_validation = scaler_x_validation.transform(X_validation)
#training
rf = RandomForestClassifier(criterion="entropy",n_estimators=200,max_features="log2",max_depth=12)
rf.fit(X_train, y_train)
#evaluation
y_pred = rf.predict(X_validation)
fone_score = f1_score(y_validation, y_pred, average="micro")
accuracy= accuracy_score(y_validation, y_pred)
print("f1 score:", fone_score)
print("accuracy score:", accuracy)
