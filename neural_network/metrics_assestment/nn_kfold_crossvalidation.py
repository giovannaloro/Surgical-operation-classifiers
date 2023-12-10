from random import randint
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras 
import numpy as np
import keras_tuner
import pandas 
import os 

#setting earlystop
early_stop = keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0.03,
    patience=10,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=15
)

#importing dataset
os.chdir('..')
df = pandas.read_csv('preprocessed_dataset/ML_MED_KFold_preprocessed.csv')
X_kf = df.iloc[:,1:17]
y_kf = df.iloc[:,17:22]
X_kf = X_kf.to_numpy()
y_kf = y_kf.to_numpy()

#crossvalidation 
seed = randint(1,100)
np.random.seed(seed)
os.chdir('neural_network/neural_models')

# define 10-fold cross validation test harness
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
run_acc = []
run_macrof1 = []
run_microf1 = []
for run in range(1,20):
    fold_acc = []
    fold_macrof1 = []
    fold_microf1 = []
    for train, test in kfold.split(X_kf, y_kf):
        # save model
        model = load_model("tuned_noweight_model.keras")
        # Fit the model
        model.fit(
            x=X_kf[train],
            y=y_kf[train],
            batch_size=10,
            epochs=100,
            callbacks=early_stop,
            shuffle=True,
            steps_per_epoch=None
            )
        # evaluate the model and save results
        scores = model.evaluate(X_kf[test], y_kf[test], verbose=0)
        fold_acc.append(scores[1])
        fold_macrof1.append(scores[2])
        fold_microf1.append(scores[3])
    run_acc.append(np.mean(fold_acc))
    run_macrof1.append(np.mean(fold_macrof1))
    run_microf1.append(np.mean(fold_microf1))
print(f"accuracy: {np.mean(run_acc)}\n")
print(f"f1score_macro: {np.mean(run_macrof1)}\n")
print(f"f1score_micro: {np.mean(run_microf1)}\n")