#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:53:22 2018

@author: mansi
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import sys

import keras

from keras.layers import Dense, Activation, MaxPool2D,MaxPooling2D, Conv2D, Flatten, Dropout
from keras.models import Sequential, Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from time import time
from sklearn.cross_validation import train_test_split

def create_model(input_dim,output_dim):
    print("Creating model....")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_dim))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))
    model.add(Activation('sigmoid'))
    print("Model is created.")
    
    return model
         
def load_data():
    filepath = os.path.join(path, "labels.txt")
    print("Path is :")
    print(filepath)
    
    if os.path.exists(filepath):
        print("Labels.txt exists.")
    else:
        print("Labels.txt does not exist -- Please check your file path.")
    
    labels = pd.read_csv(filepath, sep=" ", header=None)
    labels.columns = ["Image", "x", "y"]
    Y = labels[['y','x']].values #interchanging coordinates here because opencv reads images in the (row,column) format.
    
    X = []
    for line in labels["Image"]:
        img_path = os.path.join(path,line)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2))) #half the size
        img = img / 255.
        img = np.array(img)
        X.append(img)
    
    X = np.array(X)
    return X,Y
    
def training(model):
    print("Training....")    
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    checkpointName = os.path.join(path, "best_weights_30.h5")
    print(checkpointName)
    checkpointer = ModelCheckpoint(filepath=checkpointName, verbose=1, save_best_only=True)
    epochs = 5
    
    
    #early_stop = EarlyStopping(patience=5)
    log_dir = os.path.join(path,"logs")
    tensorboard = TensorBoard(log_dir=log_dir.format(time()))
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                     validation_split=0.2,callbacks=[checkpointer, tensorboard],
                     epochs=epochs,batch_size=64)
            
    print("Training complete.")
    return history


if __name__== "__main__":
    path = sys.argv[1]   
    X,Y = load_data()
    input_dim = X[0].shape
    output_dim = (Y[0].shape)[0]
    model = create_model(input_dim,output_dim)
    history = training(model)

