# from algorithms import *
from plot_lib import *
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
import pandas as pd
import numpy as np
import _pickle as cPickle
import code
import os

save_outcomes = False
train = False
modelsave = True
data_augmentation = False
batch_size = 16
epochs = 40
random_state = 17
n_classes = 2
split = 0.9
class_names = ["healthy", "mytotic"]
channels = 3

### Optimizer ###
lr = 0.005
momentum = 0.9
decay = 0
change_epoch = 20
lr2 = 0.001
decay2 = 0.0005

modelpath = ""

np.random.seed(seed=random_state)
data_path = "/Users/Moritz/Desktop/zeiss/data/preprocessed/"
file = "cropsize=32scanner=Ainclude_negatives=Trueratio=1hb=True"
file_path = data_path + file
cells = np.load(file_path+".npy").astype('float32')
masks = np.load(file_path+"masks.npy").astype('float32')
y=np.zeros([cells.shape[0]]).astype(int)
for i in range(0,cells.shape[0]):
    if np.max(masks[i,:,:])==1:
        y[i]=1

y = to_categorical(y)

# def naiveReshape(X, target_pixel_size):
#     X_out = np.zeros([X.shape[0], target_pixel_size, target_pixel_size, X.shape[-1]])
#     for i in range(X.shape[0]):
#         for ch in range(X.shape[-1]):
#             X_out[i,:,:,ch]=X[i,:target_pixel_size,:target_pixel_size,ch]
#     return X_out

# def get_validation_predictions(train_data, predictions_valid):
#     pv = []
#     for i in range(len(train_data)):
#         pv.append(predictions_valid[i])
#         return pv


def schedule(epoch):
    if epoch == change_epoch:
        K.set_value(model.optimizer.lr, lr2)
        K.set_value(model.optimizer.decay, decay2)
        print("Set new learning rate", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)


# reshape for network
cells = np.swapaxes(cells, 1,3)

X_train, X_test, y_train, y_test = train_test_split(cells, y, test_size=(1-split), random_state=random_state, shuffle=True)


# code.interact(local=dict(globals(), **locals()))
if data_augmentation:
    print("Data augmentation")
    X_train_l = X_train[:,:,::-1,:]
    X_train_u = X_train[:,::-1,:,:]
    X_train_lu = X_train_u[:,:,::-1,:]
    X_train = np.vstack([X_train, X_train_l, X_train_u, X_train_lu])
    y_train = np.append(y_train, np.append(y_train, np.append(y_train, y_train)))

csv_logger_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/held_back_logger.csv"
checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/held_back_checkpoint.hdf5"


#### TRAINING ####
if train:
    model = deepmytotic(channels, n_classes, lr, momentum, decay)
    #change_lr = LearningRateScheduler(schedule)
    csvlog = CSVLogger(csv_logger_path, append=True)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks = [
        #change_lr,
        csvlog,
        checkpoint,
    ]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=callbacks)
    del(model)
    model = load_model(checkpoint_path)
code.interact(local=dict(globals(), **locals()))

if not train:
    print("Loading trained model", checkpoint_path)
    model = load_model(checkpoint_path)

#### EVALUATION EX1 + Ex2 ####
predictions_train = model.predict(X_train.astype('float32'), batch_size=batch_size, verbose=2)
predictions_test = model.predict(X_test.astype('float32'), batch_size=batch_size, verbose=2)
log_loss_train = log_loss(y_train, predictions_train)
print('Score log_loss train: ', log_loss_train)
acc_train = model.evaluate(X_train.astype('float32'), y_train, verbose=0)
print("Score accuracy train: %.2f%%" % (acc_train[1]*100))

log_loss_test = log_loss(y_test, predictions_test)
print('Score log_loss test: ', log_loss_test)
acc_test = model.evaluate(X_test.astype('float32'), y_test, verbose=0)
print("Score accuracy test: %.2f%%" % (acc_test[1]*100))



plotBothConfusionMatrices(np.argmax(predictions_val, axis=1), y_test, class_names)


#### Saving Model ####
# if train & modelsave:
#     modelname = "/home/moritz_berthold/dl/cellmodels/deepflow/4_way_clean_resize" + str(resize) + "_ch_" + str(channels) + "_bs=" + str(batch_size) + \
#             "_epochs=" + str(epochs) + "_norm=" + str(data_normalization) + "_aug=" + str(entation) + "_split=" + str(split) + "_lr1=" + str(lr)  + \
#             "_momentum=" + str(momentum)  + "_decay1=" + str(decay) +  \
#             "_change_epoch=" + str(change_epoch) + "_decay2=" + str(decay2) + \
#             "_lr2=" + str(lr2)  + "_acc1=" + str(acc_train) + "_acc2=" + str(acc_test) + ".h5"
#     model.save(modelname)
#     print("saved model")


tb = pd.read_table(csv_logger_path, delimiter=",")

acc = tb["acc"]
val_acc = tb["val_acc"]

plt.plot(acc,c='r',alpha=0.5, linewidth=3)
plt.plot(val_acc,c='blue',alpha=0.5, linewidth=3)
plt.xlim([0, acc.size])
plt.ylim([0, 1])
# plt.ylim([0, np.max([np.max(loss), np.max(val_loss)])])
plt.title("Learning curves, train and val accuracies")
plt.ylabel('Accuracy')
plt.xlabel('Training epochs')
plt.show()
code.interact(local=dict(globals(), **locals()))
