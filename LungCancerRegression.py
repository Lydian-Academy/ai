import keras
import tensorflow as tf
from keras import layers
import numpy as np
import pandas as pd
import datetime
import pydot

import os
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

Lung_cancer_training_data = pd.read_csv(
    "/Users/ad/Desktop/lung_cancer.csv",
    names=["age","gender","education_years","income_level","smoker","smoking_years","cigarettes_per_day","pack_years","passive_smoking","air_pollution_index","occupational_exposure",
    "radon_exposure","family_history_cancer","copd","asthma","previous_tb","chronic_cough","chest_pain","shortness_of_breath","fatigue","bmi","oxygen_saturation",
    "fev1_x10","crp_level","xray_abnormal","exercise_hours_per_week","diet_quality","alcohol_units_per_week","healthcare_access","lung_cancer_risk"])

Lung_cancer_features = Lung_cancer_training_data.copy()
Lung_cancer_labels = Lung_cancer_features.pop('lung_cancer_risk')
Lung_cancer_features_array = np.array(Lung_cancer_features)

print("Lung_cancer_features_array")
print(Lung_cancer_features_array)
print(Lung_cancer_features_array.shape)

print("normalization")
normalize_layer = keras.layers.Normalization(axis=-1)
normalized_Lung_Cancer_data = normalize_layer.adapt(Lung_cancer_features_array)
print(normalized_Lung_Cancer_data)


lung_cancer_model = tf.keras.Sequential([
  layers.Dense(128, activation='relu', name = "layer1"),
  layers.Dense(64, activation='relu', name = "layer2"),
  layers.Dense(32, activation='relu', name = "layer3"),
  layers.Dense(1,activation='sigmoid', name = "layer4")
])


lung_cancer_model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(), metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = lung_cancer_model.fit(Lung_cancer_features_array, Lung_cancer_labels, epochs=10, batch_size=64, verbose=1, callbacks=[tensorboard_callback])

print("Summary")
print(lung_cancer_model.summary())
print("history")
print(history.history)

keras.utils.plot_model(lung_cancer_model, to_file="lung_cancer_model.png", show_shapes=True)

print("Generate prediction #1")
test_data = np.array([[51,1,13,2,1,26,12,15,0,44,1,1,0,0,0,0,1,0,0,0,24,91,26,10,1,4,3,4,2]])
prediction = lung_cancer_model.predict(test_data)
print(prediction)

print("Generate prediction #2")
test_data = np.array([[73,1,12,3,0,0,0,0,0,47,0,0,0,0,0,0,0,0,0,0,18,96,32,0,0,0,3,10,4]])
prediction2 = lung_cancer_model.predict(test_data)
print(prediction2)

print("My risk")
test_data = np.array([[70,1,21,4,0,0,0,0,0,22,0,0,0,0,0,0,0,0,0,0,27,98,35,1,0,7,3,0,5]])
prediction3 = lung_cancer_model.predict(test_data)
print(prediction3)




#print(Lung_cancer_training_data)
#print(Lung_cancer_training_data.head())
#print(Lung_cancer_training_data.shape)
#print(Lung_cancer_training_data.head())
#tensorboard --logdir /Users/ad/Desktop/logs
# to be added - normalize, model save

