# implement a neural network for classification
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.layers import Activation, Dense, Input, concatenate, UpSampling2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import numpy as np
from keras.utils import plot_model
import code
# Nick and Phils Training Details:
# The network was trained for 100 epochs using stochastic gradient descent
# with standard 318 parameters: 0.9 momentum, a fixed learning rate of 0.01
# up to epoch 85 and of 0.001 319 afterwards as well as a slightly regularizing
# weight decay of 0.0005.

# Basic Conv + BN + ReLU factory
# padding same - > same output size, padding is added according to kernel

def deepmytotic(channels, n_classes, lr, momentum, decay):
	# model = deepflow([1,2,3,4], 4, .01, .09, .0005)
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channels, 32, 32), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(UpSampling2D(size=(2,2),  dim_ordering='th'))
    model.add(UpSampling2D(size=(2,2),  dim_ordering='th'))

    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))


    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



def showModel(model, name): #only works locally showModel(model, name = "pool9")
	plot_model(model, show_shapes=True, to_file="model_visualisation/" + name + ".png")

# model.summary()
