import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras 
from tensorflow.keras.models import load_model
import numpy as np
import pandas 
import os 

#importing dataset
os.chdir('..')
df = pandas.read_csv('preprocessed_dataset/ML_MED_Dataset_train_preprocessed.csv')
X_train = df.iloc[:,1:17]
y_train = df.iloc[:,17:22]
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
df = pandas.read_csv('preprocessed_dataset/ML_MED_KFold_preprocessed.csv')
X_kf = df.iloc[:,1:17]
y_kf = df.iloc[:,17:22]
X_kf = X_kf.to_numpy()
y_kf = y_kf.to_numpy()
df = pandas.read_csv('preprocessed_dataset/ML_MED_Dataset_validation_preprocessed.csv')
X_validation = df.iloc[:,1:17]
y_validation = df.iloc[:,17:22]
X_validation = X_validation.to_numpy()
y_validation = y_validation.to_numpy()
df = pandas.read_csv('preprocessed_dataset/ML_MED_Dataset_test_preprocessed.csv')
X_test = df.iloc[:,1:17]
y_test = df.iloc[:,17:22]
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

#training
model = load_model("tuned_noweight_model.keras")
print(model.get_config())

early_stop = keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0.08,
    patience=8,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=15
)

for run in range(100):
    model = load_model("tuned_noweight_model.keras")
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=10,
        epochs=100,
        callbacks=early_stop,
        shuffle=True,
        steps_per_epoch=None
        )
    print(f"model number {run} metric: {model.evaluate(X_test, y_test)}")
    model.save(f"trained_model_{run}.h5")


