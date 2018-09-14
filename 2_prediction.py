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

import os

dataset = "A"

checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/checkpoint.hdf5"
print("Loading trained model", checkpoint_path)

model = load_model(checkpoint_path)

data_path = "/Users/Moritz/Desktop/zeiss/data/"
# Load A - Channel

# only loading the last file
file = sorted(glob.glob(data_path + dataset + "0*_v2/"))[-1]
for csv_path in sorted(glob.glob(file_path + "*.csv")):
    all_annotation_txt = np.genfromtxt(csv_path, dtype = 'str', comments='#', delimiter="',\n'", skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes')
    image = misc.imread(file_path.replace(".csv", ".png"))/255
    code.interact(local=dict(globals(), **locals()))
