#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:20:12 2018

@author: mansi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import sys
import keras
from train_phone_finder import create_model
from keras.layers import Dense, Activation, MaxPool2D,MaxPooling2D, Conv2D, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model, model_from_json
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping
from time import time
from keras.applications.vgg16 import VGG16

def test():
    
    #read a single image   
    img_path = sys.argv[1]
    parentDirectory = os.path.abspath(os.path.join(img_path, os.pardir))
    checkpointName = "best_weights_30.h5"
    checkpointFile = os.path.join(parentDirectory,checkpointName)
    
    if os.path.exists(checkpointFile):
        print("Using weights from the trained model")
    else:
        print("Model is not trained. Train the model using train_finder.py.")
    #prepare the image    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2))) #half the size
    img = img / 255.
    img = np.array(img)
    X = []
    X.append(img)
    X = np.array(X)
    
    
    test_model = create_model(X[0].shape,2)
    test_model.load_weights(checkpointFile)
    y_test = test_model.predict(X)
    print("Prediction")
    y_test = y_test[:, [1, 0]][0]
    
    return y_test

prediction = test()
print(prediction)