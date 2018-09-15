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

dataset = "A"
bounding_box_size = 32
stepSize=4
minimum_cellsize = 300
maximum_cellsize = 2800


checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/checkpoint.hdf5"
print("Loading trained model", checkpoint_path)

model = load_model(checkpoint_path)
data_path = "/Users/Moritz/Desktop/zeiss/data/"
# Load A - Channel

# only loading the last file
file_path = sorted(glob.glob(data_path + dataset + "0*_v2/"))[-1]
jaccard = []
total_true_positives = []
total_false_positives = []
total_false_negatives = []
true_totals = []
for csv_path in sorted(glob.glob(file_path + "*.csv")):
    print(csv_path)
    all_annotation_txt = np.genfromtxt(csv_path, dtype = 'str', comments='#', delimiter="',\n'", skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes')
    image = misc.imread(csv_path.replace(".csv", ".png"))/255
    mask = misc.imread(csv_path.replace(".csv", ".jpg"))/255
    cells_in_image = []
    better_mask = np.zeros_like(image[:,:,0])
    for cell_string in np.nditer(all_annotation_txt):
        cell_array = np.fromstring(np.array2string(cell_string).strip("'"), dtype=int, sep=",")
        cell_array = np.flip(np.reshape(cell_array, (-1,2)),1)
        cells_in_image.append(cell_array)
        better_mask[cell_array[:,0],cell_array[:,1]]=1

#### Prediction ####
    margin = np.round(bounding_box_size/2).astype(int)
    pad_image = np.pad(image[:,:,:3], ((margin,margin),(margin,margin),(0,0)), mode="constant", constant_values=0)
    pad_image = np.swapaxes(pad_image, 0,2)
    yields = np.zeros_like(image[:,:,0])
    for x in range(0, image.shape[0], stepSize):
        for y in range(0, image.shape[1], stepSize):
            yields[x,y] = model.predict(np.expand_dims(pad_image[:,x:x+2*margin,y:y+2*margin].astype('float32'), axis=0), batch_size=1, verbose=3)[0][1]
#### Morphological Operations ####
    yields[yields<0.99]=0
    yields[yields>=0.99]=1
    kernel = np.ones((4,4),np.uint8)
    dilation = cv2.dilate(yields,kernel,iterations = 1)
    #opening = cv2.morphologyEx(yields, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    #### Evaluation ####
    base_array, num_features = nd.label(dilation)
    indices = np.unique(base_array, return_counts=True)
    vals = []
    cell_coords = []
    for i in range(1, len(indices[1])):
        if (indices[1][i] < minimum_cellsize) or (indices[1][i] > maximum_cellsize):
            base_array[base_array==i]=0
    base_array[base_array>=1]=1
    discard_array, num_cells = nd.label(base_array)
    overlap = base_array*better_mask
    discard_array, true_positives = nd.label(overlap)
    false_positives = np.max(num_cells - true_positives,0)
    false_negatives = np.max(len(cells_in_image)-true_positives,0)
    #return average per image
    score_per_image = jaccard_similarity_score(better_mask.flatten(), yields.flatten())
    jaccard.append(score_per_image)
    total_true_positives.append(true_positives)
    total_false_positives.append(false_positives)
    total_false_negatives.append(false_negatives)
    true_totals.append(len(cells_in_image))


code.interact(local=dict(globals(), **locals()))

print("Mean Jaccard")
mean_jaccard_score = np.mean(jaccard)
print(mean_jaccard_score)

print("True Positives")
print(total_true_positives)
print(np.mean(total_true_positives))

print("False Positives")
print(total_false_positives)

print("False Negatives ")
print(total_false_negatives)










fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(yields[yields>0.2], bins=30)
yields[yields<0.99]=0
axs[0, 1].imshow(yields[:,:])
axs[1, 0].imshow(image)
axs[1, 1].imshow(mask)
plt.subplot_tool()
plt.show()

yields[yields<0.99]=0
fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.swapaxes(yields[:,:],0,1))
axs[1].imshow(better_mask)
plt.show()
