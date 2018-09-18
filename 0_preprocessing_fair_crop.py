import sys
from plot_lib import *
import numpy as np
sys.path.append('/Users/Moritz/Desktop/zeiss')
import matplotlib.pyplot as plt
seed = 17
from numpy import genfromtxt
from scipy import misc
import glob
import math
import cv2
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
import code
import os

# for unet
import keras.backend as K
K.set_image_dim_ordering('th')


from keras import backend as K
from os import environ

plot = False
random_state = 17
n_classes = 2
split = 0.9
double_channel = True
std = 0.25597024
mean = 0.5566084

save_crops = True
dataset = "A" # A or H
dataset2 = "H"
hold_back_test_data = True
factor = 16  # 32 - 64 or 16 - 128
bounding_box_size = int(2048/factor)

dimensions = 3
if double_channel:
    dimensions = 6
def draw_mask(image,  all_annotation_txt):
    mask = np.zeros_like(image[:,:,0])
    for cell_string in np.nditer(all_annotation_txt):
        cell_array = np.fromstring(np.array2string(cell_string).strip("'"), dtype=int, sep=",")
        cell_array = np.flip(np.reshape(cell_array, (-1,2)),1)
        mask[cell_array[:,0],cell_array[:,1]]=1
    return mask


def image_devider(image, mask, factor):
    image_stack=np.empty([0, bounding_box_size, bounding_box_size,dimensions])
    mask_stack=np.empty([0, bounding_box_size, bounding_box_size])
    for i in range(factor):
        for j in range(factor):
            temp_mask = mask[i*bounding_box_size:(i+1)*bounding_box_size,j*bounding_box_size:(j+1)*bounding_box_size]
            if np.max(temp_mask)>=1:
                temp_image = image[i*bounding_box_size:(i+1)*bounding_box_size,j*bounding_box_size:(j+1)*bounding_box_size,:dimensions]
                image_stack = np.concatenate((image_stack,np.expand_dims(temp_image, axis=0)), axis=0)
                mask_stack = np.concatenate((mask_stack,np.expand_dims(temp_mask, axis=0)), axis=0)
    return image_stack, mask_stack

data_path = "/Users/Moritz/Desktop/zeiss/data/"
csv_logger_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/unet_held_back_logger_size="+str(bounding_box_size)+"patch.csv"
checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/unet_held_back_checkpoint_size="+str(bounding_box_size)+"patch.hdf5"

# Load A - Channel
all_cells = np.empty([0, bounding_box_size, bounding_box_size, dimensions])
all_masks = np.empty([0, bounding_box_size, bounding_box_size])
a_channel_annotations = []
if hold_back_test_data:
    filelist = sorted(glob.glob(data_path + dataset + "0*_v2/"))[:-1]
else:
    filelist = sorted(glob.glob(data_path + dataset + "0*_v2/"))
for file_path in filelist:
    for csv_path in sorted(glob.glob(file_path + "*.csv")):
        all_annotation_txt = np.genfromtxt(csv_path, dtype = 'str', comments='#', delimiter="',\n'", skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes')
        image = misc.imread(csv_path.replace(".csv", ".png"))[:,:,:3]/255
        if double_channel:
            image2 = misc.imread(csv_path.replace(".csv", ".png").replace(dataset, dataset2))[:,:,:3]/255
            image2 = cv2.resize(image2, (image.shape[0], image.shape[1]), interpolation = cv2.INTER_LINEAR)
            image = np.concatenate((image, image2), axis=-1)
        cells = np.empty([0, bounding_box_size, bounding_box_size, dimensions])
        masks = np.empty([0, bounding_box_size, bounding_box_size])
        mask = draw_mask(image, all_annotation_txt)
        mask = mask[18:2066,18:2066]
        image = image[18:2066,18:2066,:]
        image_sections, mask_sections = image_devider(image, mask, factor)
        all_cells = np.concatenate((all_cells, image_sections), axis=0)
        all_masks = np.concatenate((all_masks, mask_sections), axis=0)
    print(file_path)


if save_crops:
    filename = data_path + "preprocessed/" + "fair_cropsize=" + str(bounding_box_size)+ "scanner=" +dataset+"hb="+str(hold_back_test_data) + "double_channel="+str(double_channel)
    np.save(filename+ ".npy", all_cells, allow_pickle=True, fix_imports=True)
    np.save(filename+ "masks.npy", all_masks, allow_pickle=True, fix_imports=True)





code.interact(local=dict(globals(), **locals()))
if plot:
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(all_cells[-3,:,:,:])
    axs[1].imshow(all_masks[-3,:,:])
    plt.show()
