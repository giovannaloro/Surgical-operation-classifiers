import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras 
import numpy as np
import keras_tuner
from tensorflow.keras.datasets import mnist
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

#hyperparameter tuning 
def build_model(hp):

    model = keras.Sequential()

    input_dropout = hp.Float("indrop", min_value=0.1, max_value=0.5, sampling="log")
    input_layer = keras.layers.Dense(16, input_shape=[16],activation='relu')
    model.add(keras.layers.Dropout(input_dropout))

    inner_layers=hp.Int("inner_layers", min_value=1, max_value=3, step=1)
    inner_neurons=hp.Int("inner_neurons", min_value=8, max_value=32, step=4)
    for x in range(inner_layers):
        model.add(keras.layers.Dense(inner_neurons,activation='relu'))

    output_layer = keras.layers.Dense(5, activation='softmax')
    model.add(output_layer)

    learning_rate = hp.Float("lr", min_value=0.001 ,max_value=0.3, sampling="log" )
    momentum = hp.Float("momentum", min_value=0.001 ,max_value=0.5, sampling="log" )
    model.compile(optimizer=keras.optimizers.SGD( learning_rate=learning_rate, momentum=momentum), loss ='categorical_crossentropy', metrics=["categorical_accuracy",tfa.metrics.F1Score(average='macro',num_classes=5),tfa.metrics.F1Score(average='macro',num_classes=5)])

    return model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="categorical_accuracy",
    max_trials=100,
    executions_per_trial=10,
    overwrite=True,
    directory="hp_nn_search",
    project_name="Surgical_classification",
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_validation, y_validation))

# Get the top  hyperparameters.
best_hps = tuner.get_best_hyperparameters(5)
# Build the model with the best hp.
print(best_hps[0])
model = build_model(best_hps[0])
print(best_hps[0])
# Save the model.
model.save("tuned_noweight_model.keras")

