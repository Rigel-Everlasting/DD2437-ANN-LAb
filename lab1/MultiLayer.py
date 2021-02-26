#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
# @author:Clarky Clark Wang
# @license: Apache Licence 
# @file: MultiLayer.py 
# @time: 2021/01/23
# @contact: wangz@kth,se
# @software: PyCharm 
# Import Libs and Let's get started, shall we?
import time
from DataGen import *
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization


def split_data(input, output, ratio):
    return train_test_split(input, output, test_size=ratio, random_state=2021)


def model(x_train, x_test, y_train, one_hidden, two_hidden, learning_rate, momentum, epochs, reg_flag = True):
    features,_ = x_train.shape
    model = Sequential()
    model.add(Dense(3, input_shape=(features,)))
    if reg_flag:
        model.add(Dropout(0.2))
    model.add(Dense(one_hidden))
    if reg_flag:
        model.add(Dropout(0.2))
    model.add(Dense(two_hidden))
    model.add(Dense(1))

    sgd = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', patience=10)

    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2, callbacks=[es])
    elapsed_time = time.time() - start_time
    yhat = model.predict(x_test)
    weights, biases = model.layers[0].get_weights()

    return model, yhat, history.history['val_loss'][len(history.history['val_loss']) - 1], weights, elapsed_time


if __name__ == '__main__':
    data = DataGenerator()
    x = data.mackey_glass()
    input, output = data.time_series(x)
    print(input.shape)
    X_train, X_test, y_train, y_test = split_data(input, output, 0.25)
    x_train, x_test = X_train.T, X_test.T
    model, yhat, history, weight, lasting = model(x_train, x_test, y_train, 4, 6, 0.001, 0.99, 40)
    print(weight)
