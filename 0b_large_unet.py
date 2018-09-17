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


train = True
modelsave = True
data_augmentation = True
batch_size = 16
epochs = 1
random_state = 17
n_classes = 2
split = 0.9
class_names = ["healthy", "mytotic"]
double_channel = True
std = 0.25597024
mean = 0.5566084


plot = False
save_crops = True
dataset = "A" # A or H
dataset2 = "H"
include_negatives=True
sample_ratio=2
hold_back_test_data = True
double_channel=False
factor = 8 # 521
bounding_box_size = int(2048/factor)

dimensions = 3
if double_channel:
    dimensions = 4
def draw_mask(image,  all_annotation_txt):
    mask = np.zeros_like(image[:,:,0])
    for cell_string in np.nditer(all_annotation_txt):
        cell_array = np.fromstring(np.array2string(cell_string).strip("'"), dtype=int, sep=",")
        cell_array = np.flip(np.reshape(cell_array, (-1,2)),1)
        mask[cell_array[:,0],cell_array[:,1]]=1
    return mask


def image_devider(image, factor):
    image_stack = np.reshape(image[:,:,:3], (factor**2,int(image.shape[0]/factor),int(image.shape[1]/factor),3), order="F")
    return image_stack

def mask_devider(image, factor):
    image_stack = np.reshape(image, (factor**2,int(image.shape[0]/factor),int(image.shape[1]/factor)), order="F")
    return image_stack


data_path = "/Users/Moritz/Desktop/zeiss/data/"
csv_logger_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/unet_held_back_logger_256patch.csv"
checkpoint_path = "/Users/Moritz/Desktop/zeiss/resources/checkpoints/unet_held_back_checkpoint_256patch.hdf5"

# Load A - Channel
all_cells = np.empty([0, bounding_box_size, bounding_box_size, dimensions])
all_masks = np.empty([0, bounding_box_size, bounding_box_size])
a_channel_annotations = []
model = unet_large(nClasses=1, input_width=int(bounding_box_size), input_height=int(bounding_box_size), nChannels=3)
csvlog = CSVLogger(csv_logger_path, append=True)
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks = [
    #change_lr,
    csvlog,
    checkpoint,
]

if hold_back_test_data:
    filelist = sorted(glob.glob(data_path + dataset + "0*_v2/"))[:-1]
else:
    filelist = sorted(glob.glob(data_path + dataset + "0*_v2/"))
for file_path in filelist:
    for csv_path in sorted(glob.glob(file_path + "*.csv")):
        all_annotation_txt = np.genfromtxt(csv_path, dtype = 'str', comments='#', delimiter="',\n'", skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes')
        image = misc.imread(csv_path.replace(".csv", ".png"))/255
        if double_channel:
            image2 = misc.imread(csv_path.replace(".csv", ".png").replace(dataset, dataset2))/255
            image2 = np.mean(cv2.resize(image2, (image.shape[0], image.shape[1]), interpolation = cv2.INTER_LINEAR), axis=-1)
            image2 = np.expand_dims(image2, axis=-1)
            image = np.concatenate((image, image2), axis=-1)
        mask = draw_mask(image, all_annotation_txt)
        mask = mask[18:2066,18:2066]
        image = image[18:2066,18:2066,:]
        mask_sections = mask_devider(mask, factor)
        image_sections = image_devider(image, factor)
        all_cells = np.concatenate((all_cells, image_sections), axis=0)
        all_masks = np.concatenate((all_masks, mask_sections), axis=0)


        if data_augmentation:
            # augmentation shift
            print("Data augmentation")
            X_train_l = all_cells[:,:,::-1,:]
            X_train_u = all_cells[:,::-1,:,:]
            X_train_lu = all_cells[:,:,::-1,:]
            X_train = np.vstack([all_cells, X_train_l, X_train_u, X_train_lu])
            y_train_l = all_masks[:,::-1,:]
            y_train_u = all_masks[::-1,:,:]
            y_train_lu = all_masks[:,::-1,:]
            y_train = np.vstack([all_masks, y_train_l, y_train_u, y_train_lu])

        X_train = (X_train-mean)/std
        X_train = np.swapaxes(X_train, 1,3)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=(1-split), random_state=random_state, shuffle=True)
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=3, validation_data=(X_test, y_test), callbacks=callbacks)
    code.interact(local=dict(globals(), **locals()))



        # crop down to 512 by 512
        # data data_augment
        # train unet on 512 by 512
        # datagen = ImageDataGenerator(
        #     featurewise_center=True,
        #     featurewise_std_normalization=True,
        #     rotation_range=20,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     horizontal_flip=True)
        #
        # # compute quantities required for featurewise normalization
        # # (std, mean, and principal components if ZCA whitening is applied)
        # datagen.fit(x_train)
        #
        # # fits the model on batches with real-time data augmentation:
        # model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
        #                     steps_per_epoch=len(x_train) / 32, epochs=epochs)































if save_crops:
    filename = data_path + "preprocessed/" + "cropsize=" + str(bounding_box_size)+ "scanner=" +dataset+"include_negatives="+str(include_negatives)+"ratio="+str(sample_ratio)+"hb="+str(hold_back_test_data) + "double_channel_mean="+str(double_channel)
    np.save(filename+ ".npy", all_cells, allow_pickle=True, fix_imports=True)
    np.save(filename+ "masks.npy", all_masks, allow_pickle=True, fix_imports=True)

        #print(len(cells_in_image))
    #    a_channel_annotations.append(np.asarray(cells_in_image))
    #for image_path in glob.glob(file_path + "*.jpg"):
    #    image = misc.imread(image_path)/255
    #    a_channel_images = np.vstack((a_channel_images, np.expand_dims(image, axis=0)))
total_annotated_cells = sum([len(x) for x in a_channel_annotations])
cells = [item for sublist in a_channel_annotations for item in sublist]
x_diameters = [(np.max(coords[:,0])-np.min(coords[:,0])) for coords in cells]
y_diameters = [(np.max(coords[:,1])-np.min(coords[:,1])) for coords in cells]
sizes = [len(coords) for coords in cells]

if plot:
    plotHistogram(x_diameters,'Pixel','Häufigkeit',"X-Durchmesser")
    plotHistogram(y_diameters,'Pixel','Häufigkeit',"Y-Durchmesser")
    plotHistogram(sizes,'Total-Pixel-Count','Häufigkeit',"Cell-Size")
    plt.show()
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(all_cells[10,:,:,0])
    axs[1].imshow(all_cells[10,:,:,-1])
    plt.show()
# Load H - Channel
