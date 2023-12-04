
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


# import contest.agents.IntelArtif_P4_U199140_U185166.myTeam


# agents\IntelArtif-P4_U199140_U185166
# python agents\IntelArtif-P4_U199140_U185166\ModelSetup.py

# python agents\IntelArtif_P4_U199140_U185166\ModelSetup.py

print("Imports done")


log_path = os.path.join("agents", "IntelArtif_P4_U199140_U185166", "model")

print(log_path)


model = Sequential([Input(shape=(13+16*32,))])
 # 525
model.add(Dense(units=48, activation='softmax'))
model.add(Dense(units=64, activation='softmax'))
model.add(Dense(units=64, activation='softmax'))
model.add(Dense(units=32, activation='softmax'))
model.add(Dense(units=5, activation='softmax'))
# 5 neurons in final layer, for each action (4 dirs + stop)

# optimizer='sgd'
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

model.summary()


print("Model Compiled")

full_path = os.path.join(log_path, "tf_pacman_model")

# model.save('tf_pacman_model')

save_model(model=model, filepath=full_path, overwrite=True)







