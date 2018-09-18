# from algorithms import *
from models import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.models import load_model
from sklearn.metrics import log_loss
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras.losses
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
import pandas as pd
import code
import os

# for unet
import keras.backend as K
K.set_image_dim_ordering('th')


from keras import backend as K
from os import environ

fair = True
train = True
modelsave = True
data_augmentation = True
batch_size = 32
epochs = 60
change_epoch = 40
random_state = 17
n_classes = 2
split = 0.9
class_names = ["healthy", "mytotic"]
double_channel = True
thresholding_probabilities = True

lr2 = 5e-6
decay2 = 0.0005

modelpath = ""


data_path = "/Users/Moritz/Desktop/zeiss/data/preprocessed/"
file = "fair_cropsize=128scanner=Ainclude_negatives=Truehb=Truedouble_channel=False"
file_path = data_path + file
cells = np.load(file_path+".npy").astype('float32')
masks = np.load(file_path+"masks.npy").astype('float32')


print("Cell Dimensions")
print(cells.shape)
def schedule(epoch):
    if epoch == change_epoch:
        K.set_value(model.optimizer.lr, lr2)
        K.set_value(model.optimizer.decay, decay2)
        print("Set new learning rate", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)


# reshape for network
cells = np.swapaxes(cells, 1,3)
if fair:
    masks=np.swapaxes(masks,1,2)

X_train, X_test, y_train, y_test = train_test_split(cells, masks, test_size=(1-split), random_state=random_state, shuffle=True)


mean = np.mean(X_train)# (0)[np.newaxis,:]  # mean for data centering
std = np.std(X_train)  # std for data normalization
X_train -= mean
X_train /= std
X_test -= mean
X_test /= std




if data_augmentation:
    print("Data augmentation")
    X_train_l = X_train[:,:,::-1,:]
    X_train_u = X_train[:,::-1,:,:]
    X_train_lu = X_train_u[:,:,::-1,:]
    X_train = np.vstack([X_train, X_train_l, X_train_u, X_train_lu])
    y_train_l = y_train[:,::-1,:]
    y_train_u = y_train[::-1,:,:]
    y_train_lu = y_train_u[:,::-1,:]
    y_train = np.vstack([y_train, y_train_l, y_train_u, y_train_lu])

csv_logger_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/fair_unet_held_back_logger_single_128.csv"
checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/fair_unet_held_back_checkpoint_single_128.hdf5"


#### TRAINING ####
if train:
    model = unet_large(nClasses=1, input_width=cells.shape[-1], input_height=cells.shape[-2], nChannels=cells.shape[-3])
    change_lr = LearningRateScheduler(schedule)
    csvlog = CSVLogger(csv_logger_path, append=True)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks = [
        #change_lr,
        csvlog,
        checkpoint,
    ]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=callbacks)
    del(model)
    keras.losses.dice_coef_loss = dice_coef_loss
    keras.metrics.dice_coef = dice_coef
    model = load_model(checkpoint_path)


if not train:
    keras.losses.dice_coef_loss = dice_coef_loss
    keras.metrics.dice_coef = dice_coef
    print("Loading trained model", checkpoint_path)
    model = load_model(checkpoint_path)


code.interact(local=dict(globals(), **locals()))


#### EVALUATION EX1 + Ex2 ####
predictions_train = model.predict(X_train.astype('float32'), batch_size=batch_size, verbose=2)*std + mean
predictions_test = model.predict(X_test.astype('float32'), batch_size=batch_size, verbose=2)*std + mean

if thresholding_probabilities:
    thres = 0.65
    predictions_test[predictions_test>thres]=1
    predictions_test[predictions_test<=thres]=0

fig, axs = plt.subplots(3, 3)
axs[0, 0].imshow(X_test[4,0,:,:])
axs[0, 1].imshow(predictions_test[4,0,:,:])
axs[0, 2].imshow(y_test[4,:,:])
axs[1, 0].imshow(X_test[7,0,:,:])
axs[1, 1].imshow(predictions_test[7,0,:,:])
axs[1, 2].imshow(y_test[7,:,:])
axs[2, 0].imshow(X_test[14,0,:,:])
axs[2, 1].imshow(predictions_test[14,0,:,:])
axs[2, 2].imshow(y_test[14,:,:])
plt.subplot_tool()
plt.show()







plt.imshow(predictions_test[13,0,:,:])
plt.show()


tb = pd.read_table(csv_logger_path, delimiter=",")

acc = tb["loss"]
val_acc = tb["val_loss"]

plt.plot(acc,c='r',alpha=0.5, linewidth=3)
plt.plot(val_acc,c='blue',alpha=0.5, linewidth=3)
plt.xlim([0, acc.size])
plt.ylim([0, 1])
# plt.ylim([0, np.max([np.max(loss), np.max(val_loss)])])
plt.title("Learning curves, train and val loss")
plt.ylabel('Loss')
plt.xlabel('Training epochs')
plt.show()
code.interact(local=dict(globals(), **locals()))
