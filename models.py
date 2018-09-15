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
# from keras.utils import plot_model
# from keras.layers import Reshape
# from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
# from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
# from keras.layers.normalization import BatchNormalization
# from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
# from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
# from keras.layers.convolutional import Convolution1D, MaxPooling1D
# from keras.layers.recurrent import LSTM
# from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import Adam , SGD
# from keras.layers.embeddings import Embedding
# from keras.utils import np_utils
# from keras.regularizers import ActivityRegularizer
from keras import backend as K
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


# def unet (nClasses , optimizer=None , input_width=360 , input_height=480 , nChannels=1 ):
#
#     inputs = Input((nChannels, input_height, input_width)) # 32
#     conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(inputs)
#     conv1 = Dropout(0.2)(conv1)
#     conv1 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#
#     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(pool1) #64
#     conv2 = Dropout(0.2)(conv2)
#     conv2 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#
#     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(pool2) #128
#     conv3 = Dropout(0.2)(conv3)
#     conv3 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv3)
#
#     up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1) #64
#     conv4 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(up1)
#     conv4 = Dropout(0.2)(conv4)
#     conv4 = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(conv4)
#
#     up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1) #32
#     conv5 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(up2)
#     conv5 = Dropout(0.2)(conv5)
#     conv5 = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(conv5)
#
#     conv6 = Convolution2D(nClasses, 1, 1, activation='relu',border_mode='same')(conv5)
#     conv6 = core.Reshape((nClasses,input_height*input_width))(conv6)
#     conv6 = core.Permute((2,1))(conv6)
#     conv7 = core.Activation('softmax')(conv6)
#     model = Model(input=inputs, output=conv7)
#
#     if not optimizer is None:
# 	    model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )
#     return model
#
#
#




def showModel(model, name): #only works locally showModel(model, name = "pool9")
	plot_model(model, show_shapes=True, to_file="model_visualisation/" + name + ".png")






# model.summary()
