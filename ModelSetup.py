
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


import contest.agents.IntelArtif_P4_U199140_U185166.myTeam


# agents\IntelArtif-P4_U199140_U185166
# python agents\IntelArtif-P4_U199140_U185166\ModelSetup.py

# python agents\IntelArtif_P4_U199140_U185166\ModelSetup.py

print("Imports done")


log_path = os.path.join("agents", "IntelArtif_P4_U199140_U185166", "model")

print(log_path)


model = Sequential()
model.add(Dense(units=48, activation='softmax', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='softmax'))
model.add(Dense(units=64, activation='softmax'))
model.add(Dense(units=32, activation='softmax'))
model.add(Dense(units=5, activation='softmax'))
# 5 neurons in final layer, for each action (4 dirs + stop)

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')

print("Model Compiled")

# model.save('tf_pacman_model')








