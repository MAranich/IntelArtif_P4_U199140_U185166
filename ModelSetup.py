
import os # operating system lib
import random
import numpy as np


# print(os.getcwd())


from contest.captureAgents import CaptureAgent
from contest.game import Directions

# import contest.util
# from contest.util import nearestPoint


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.saving import save_model
import tensorflow as tf


# import contest.agents.IntelArtif_P4_U199140_U185166.myTeam


# agents\IntelArtif-P4_U199140_U185166
# python agents\IntelArtif-P4_U199140_U185166\ModelSetup.py

# python agents\IntelArtif_P4_U199140_U185166\ModelSetup.py

print("Imports done")

print("TensorFlow version:", tf.__version__)


log_path = os.path.join("agents", "IntelArtif_P4_U199140_U185166", "model")

print(log_path)

# print((525,), (525), (None, 1))  # 13+16*32

# model = Sequential([Input(shape= 525)])
model = Sequential([
    Dense(units=48, activation='softmax', input_shape=(525, 1)), 
    Dense(units=64, activation='softmax'), 
    Dense(units=64, activation='softmax'), 
    Dense(units=32, activation='softmax'), 
    Dense(units=5, activation='softmax')
])

r"""
model.add(Dense(units=48, activation='softmax'))
model.add(Dense(units=64, activation='softmax'))
model.add(Dense(units=64, activation='softmax'))
model.add(Dense(units=32, activation='softmax'))
model.add(Dense(units=5, activation='softmax'))
"""
# 5 neurons in final layer, for each action (4 dirs + stop)

# optimizer='sgd'  # mse = mean squared error
# model.compile(loss=tf.keras.losses.mse, optimizer='adam', metrics=["mse"])
model.compile(loss=tf.keras.losses.mse, optimizer='adam')

model.build()

model.summary()


print("Model Compiled. ")

full_path = os.path.join(log_path, "tf_pacman_model")

# model.save('tf_pacman_model')

save_model(model=model, filepath=full_path, overwrite=True)

print("Model Saved. ")






