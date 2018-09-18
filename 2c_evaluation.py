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
from sklearn.metrics import jaccard_similarity_score
import scipy.ndimage as nd
import cv2
import numpy as np
import _pickle as cPickle
import code
import glob
from scipy import misc

import os

plot = True
dataset = "A"
dataset2 = "H"
bounding_box_size = 64
minimum_cellsize = 300
maximum_cellsize = 2800
double_channel=False
dimensions = 3
thresholding_probabilities = True
fair = True
if double_channel:
    dimensions = 6



#checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/unet_held_back_checkpoint.hdf5"
# checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/unet_held_back_checkpoint_double.hdf5"
# checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/unet_held_back_checkpoint_double_decay.hdf5"
checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/fair_unet_held_back_checkpoint_single_64.hdf5"

print("Loading trained model", checkpoint_path)

thres = 0.65
std = 0.25597024
mean = 0.5566084


smooth=1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

import keras
keras.losses.dice_coef_loss = dice_coef_loss
keras.metrics.dice_coef = dice_coef
print("Loading trained model", checkpoint_path)
model = load_model(checkpoint_path)

model = load_model(checkpoint_path)
data_path = "/Users/Moritz/Desktop/zeiss/data/"
# Load A - Channel

# only loading the last file
file_path = sorted(glob.glob(data_path + dataset + "0*_v2/"))[-1]
dice_coef_loss_list = []
total_true_positives = []
total_false_positives = []
total_false_negatives = []
true_totals = []
for csv_path in sorted(glob.glob(file_path + "*.csv")):
    print(csv_path)
    all_annotation_txt = np.genfromtxt(csv_path, dtype = 'str', comments='#', delimiter="',\n'", skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes')
    image = misc.imread(csv_path.replace(".csv", ".png"))/255
    if double_channel:
        image2 = misc.imread(csv_path.replace(".csv", ".png").replace(dataset, dataset2))/255
        image2 = cv2.resize(image2, (image.shape[0], image.shape[1]), interpolation = cv2.INTER_LINEAR)
        image = np.concatenate((image, image2), axis=-1)
    mask = misc.imread(csv_path.replace(".csv", ".jpg"))/255
    cells_in_image = []
    better_mask = np.zeros_like(image[:,:,0])
    for cell_string in np.nditer(all_annotation_txt):
        cell_array = np.fromstring(np.array2string(cell_string).strip("'"), dtype=int, sep=",")
        cell_array = np.flip(np.reshape(cell_array, (-1,2)),1)
        cells_in_image.append(cell_array)
        better_mask[cell_array[:,0],cell_array[:,1]]=1
        del cell_array

#### Prediction ####
    margin = np.round(bounding_box_size/2).astype(int)
    pad_image = np.pad(image[:,:,:dimensions], ((margin,margin),(margin,margin),(0,0)), mode="constant", constant_values=0)
    pad_image = (np.swapaxes(pad_image, 0,2)- mean)/std

    yields = np.zeros_like(pad_image[0,:,:])
    for x in range(0, image.shape[0], margin):
        for y in range(0, image.shape[1], margin):
            temp_yield = model.predict(np.expand_dims(pad_image[:,x:x+2*margin,y:y+2*margin].astype('float32'), axis=0), batch_size=1, verbose=3)[0][0]
            # if thresholding_probabilities:
            #     temp_yield[temp_yield<thres]=0
            #     temp_yield[temp_yield>=thres]=1
            yields[x:x+2*margin,y:y+2*margin] = np.mean((temp_yield, yields[x:x+2*margin,y:y+2*margin]), axis=0)

    thres=.8
    if thresholding_probabilities:
        yields[yields<thres]=0
        yields[yields>=thres]=1
    #
    # if plot:
    #     fig, axs = plt.subplots(1, 2)
    #     axs[0].imshow(yields)
    #     axs[1].imshow(image[:,:,:3])
    #     plt.show()
    image = image * std + mean
    yields = yields[margin:-margin,margin:-margin]




    #### Morphological Operations ####
    # kernel = np.ones((stepSize+2,stepSize+2),np.uint8)
    # dilation = cv2.dilate(yields,kernel,iterations = 1)
    #opening = cv2.morphologyEx(yields, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    dilation = yields
    #### Evaluation ####
    base_array, num_features = nd.label(dilation)
    indices = np.unique(base_array, return_counts=True)
    vals = []
    cell_coords = []
    discarded = 0
    for i in range(1, len(indices[1])):
        if (indices[1][i] < minimum_cellsize) or (indices[1][i] > maximum_cellsize):
            base_array[base_array==i]=0
            discarded +=1
    print("Discarded Cells per image")
    print(discarded)
    base_array[base_array>=1]=1
    discard_array, num_cells = nd.label(base_array)
    better_mask = np.swapaxes(better_mask,0,1)
    overlap = base_array*better_mask
    discard_array, true_positives = nd.label(overlap)
    false_positives = np.max(num_cells - true_positives,0)
    false_negatives = np.max(len(cells_in_image)-true_positives,0)
    #return average per image
    score_per_image = dice_coef_loss(better_mask, yields)
    dice_coef_loss_list.append(score_per_image)
    total_true_positives.append(true_positives)
    total_false_positives.append(false_positives)
    total_false_negatives.append(false_negatives)
    true_totals.append(len(cells_in_image))

    if plot:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(image[:,:,:3])
        axs[1, 0].imshow(image[:,:,:3])
        axs[0, 1].imshow(base_array)
        axs[1, 1].imshow(better_mask)
        plt.show()
    code.interact(local=dict(globals(), **locals()))

code.interact(local=dict(globals(), **locals()))

print("Mean Dice Loss")
mean_dice_loss = np.mean(dice_coef_loss_list)
print(mean_dice_loss)

print("True Positives")
print(total_true_positives)

print("False Positives")
print(total_false_positives)

print("False Negatives ")
print(total_false_negatives)

print("True Totals ")
print(true_totals)



yields[yields<0.99]=0
fig, axs = plt.subplots(1, 2)
axs[0].imshow(better_mask)
axs[1].imshow(base_array)
plt.show()
