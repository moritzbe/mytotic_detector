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
import numpy as np
import _pickle as cPickle
import code
import glob
from scipy import misc

import os

dataset = "A"
bounding_box_size = 32


checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/checkpoint.hdf5"
print("Loading trained model", checkpoint_path)

model = load_model(checkpoint_path)
data_path = "/Users/Moritz/Desktop/zeiss/data/"
# Load A - Channel

def create_better_mask()


# only loading the last file
file_path = sorted(glob.glob(data_path + dataset + "0*_v2/"))[-1]
for csv_path in sorted(glob.glob(file_path + "*.csv")):
    all_annotation_txt = np.genfromtxt(csv_path, dtype = 'str', comments='#', delimiter="',\n'", skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes')
    image = misc.imread(csv_path.replace(".csv", ".png"))/255
    mask = misc.imread(csv_path.replace(".csv", ".jpg"))/255
    better_mask = create_better_mask()

margin = np.round(bounding_box_size/2).astype(int)
pad_image = np.pad(image[:,:,:3], ((margin,margin),(margin,margin),(0,0)), mode="constant", constant_values=0)
pad_image = np.swapaxes(pad_image, 0,2)
yields = np.zeros_like(image[:,:,0])

stepSize=4
for x in range(0, image.shape[0], stepSize):
    for y in range(0, image.shape[1], stepSize):
        yields[x,y] = model.predict(np.expand_dims(pad_image[:,x:x+2*margin,y:y+2*margin].astype('float32'), axis=0), batch_size=1, verbose=3)[0][1]


fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(yields[yields>0.2], bins=30)
yields[yields<0.98]=0
axs[0, 1].imshow(yields[:,:])
axs[1, 0].imshow(image)
axs[1, 1].imshow(mask)
plt.subplot_tool()
plt.show()


fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.swapaxes(yields[:,:],0,1))
axs[1].imshow(image)
plt.show()

code.interact(local=dict(globals(), **locals()))
