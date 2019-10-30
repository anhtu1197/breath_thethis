import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, LSTM

class LSTM_MODEL(object):
    @staticmethod
    def build(data_input_shape, classes):
        model = Sequential()
        # Configuring the parameters

        #1
        # model.add(LSTM(32, input_shape=data_input_shape))
        # # Adding a dropout layer
        # model.add(Dropout(0.5))
        # # Adding a dense output layer with sigmoid activation
        # model.add(Dense(classes, activation='sigmoid'))

        # #2
        # model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=data_input_shape))
        # model.add(LSTM(units=32,  dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        # model.add(Dense(units=classes, activation="softmax"))

        # #3
        model.add(LSTM(32,return_sequences=True,input_shape=data_input_shape))
        # Adding a dropout layer
        model.add(Dropout(0.5))

        model.add(LSTM(28,input_shape=data_input_shape))
        # Adding a dropout layer
        model.add(Dropout(0.6))
        # Adding a dense output layer with sigmoid activation
        model.add(Dense(classes, activation='sigmoid'))


        model.summary()



        # Keras optimizer defaults:
        # Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
        # RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
        # SGD    : lr=0.01,  momentum=0.,                             decay=0.
        opt = Adam(lr=0.1)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model
