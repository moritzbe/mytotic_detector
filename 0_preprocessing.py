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


plot = False
bounding_box_size = 64
save_crops = True
dataset = "A" # A or H
include_negatives=True
sample_ratio=1
hold_back_test_data = True

def crop_image(image, cell_center, bounding_box_size, cell_array):
    margin = np.round(bounding_box_size/2).astype(int)
    pad_image = np.pad(image[:,:,:3], ((margin,margin),(margin,margin),(0,0)), mode="constant", constant_values=0)
    crop = pad_image[cell_center[0]:cell_center[0]+2*margin, cell_center[1]:cell_center[1]+2*margin,:]
    mask = np.zeros_like(crop[:,:,0])

    left = cell_array[:,0] > cell_center[0]-margin
    right = cell_array[:,0] < cell_center[0]+margin
    low = cell_array[:,1] > cell_center[1]-margin
    up = cell_array[:,1] < cell_center[1]+margin
    condition = left*right*low*up
    cropped_array = cell_array[condition]
    mask[cropped_array[:,0]+margin-cell_center[0],cropped_array[:,1]+margin-cell_center[1]]=1
    del pad_image, image
    return crop, mask

def crop_neg(image, cell_center, bounding_box_size):
    margin = np.round(bounding_box_size/2).astype(int)
    pad_image = np.pad(image[:,:,:3], ((margin,margin),(margin,margin),(0,0)), mode="constant", constant_values=0)
    neg = pad_image[cell_center[0]:cell_center[0]+2*margin, cell_center[1]:cell_center[1]+2*margin,:]
    neg_mask = np.zeros_like(neg[:,:,0])
    del pad_image, image
    return neg, neg_mask


data_path = "/Users/Moritz/Desktop/zeiss/data/"
# Load A - Channel
a_channel_cells = np.empty([0, bounding_box_size, bounding_box_size, 3])
a_channel_masks = np.empty([0, bounding_box_size, bounding_box_size])
a_channel_annotations = []
if hold_back_test_data:
    filelist = sorted(glob.glob(data_path + dataset + "0*_v2/"))[:-1]
else:
    filelist = sorted(glob.glob(data_path + dataset + "0*_v2/"))
for file_path in filelist:
    for csv_path in sorted(glob.glob(file_path + "*.csv")):
        all_annotation_txt = np.genfromtxt(csv_path, dtype = 'str', comments='#', delimiter="',\n'", skip_header=0, skip_footer=0, converters=None, missing_values=None, filling_values=None, usecols=None, names=None, excludelist=None, deletechars=None, replace_space='_', autostrip=False, case_sensitive=True, defaultfmt='f%i', unpack=None, usemask=False, loose=True, invalid_raise=True, max_rows=None, encoding='bytes')
        image = misc.imread(csv_path.replace(".csv", ".png"))/255
        cells_in_image = []
        for cell_string in np.nditer(all_annotation_txt):
            cell_array = np.fromstring(np.array2string(cell_string).strip("'"), dtype=int, sep=",")
            cell_array = np.flip(np.reshape(cell_array, (-1,2)),1)
            cell_center = np.round(np.mean(cell_array, axis=0)).astype(int)
            cells_in_image.append(cell_array)
            crop, mask = crop_image(image, cell_center, bounding_box_size, cell_array)
            a_channel_cells = np.vstack((a_channel_cells, np.expand_dims(crop, axis=0)))
            a_channel_masks = np.vstack((a_channel_masks, np.expand_dims(mask, axis=0)))
            if plot:
                plt.imshow(crop)
                plt.show()
                plt.imshow(mask)
                plt.show()
            del(mask, crop)
        a_channel_annotations.append(cells_in_image)

if include_negatives:
    negs = np.empty([0, bounding_box_size, bounding_box_size, 3])
    negs_masks = np.empty([0, bounding_box_size, bounding_box_size])
    if hold_back_test_data:
        filelist = sorted(glob.glob(data_path + dataset + "0*_v2/*.png"))[:-1]
    else:
        filelist = sorted(glob.glob(data_path + dataset + "0*_v2/*.png"))
    for image_path in filelist:
            image = misc.imread(image_path)/255
            negs_in_image = []
            n = math.ceil(320*sample_ratio / 80) * 2 # round to even number
            neg_centers = np.random.randint(0, 2048, n*2)
            neg_centers = np.flip(np.reshape(neg_centers, (-1,2)),1)
            for i in range(neg_centers.shape[0]):
                neg_center = neg_centers[i,:]
                neg, neg_mask = crop_neg(image, neg_center, bounding_box_size)
                negs = np.vstack((negs, np.expand_dims(neg, axis=0)))
                negs_masks = np.vstack((negs_masks, np.expand_dims(neg_mask, axis=0)))
                if plot:
                    plt.imshow(neg)
                    plt.show()
                    plt.imshow(neg_mask)
                    plt.show()
                del(neg_mask, neg)
    a_channel_cells = np.vstack((a_channel_cells, negs))
    a_channel_masks = np.vstack((a_channel_masks, negs_masks))



if save_crops:
    filename = data_path + "preprocessed/" + "cropsize=" + str(bounding_box_size)+ "scanner=" +dataset+"include_negatives="+str(include_negatives)+"ratio="+str(sample_ratio)+"hb="+str(hold_back_test_data)
    np.save(filename+ ".npy", a_channel_cells, allow_pickle=True, fix_imports=True)
    np.save(filename+ "masks.npy", a_channel_masks, allow_pickle=True, fix_imports=True)


        #print(len(cells_in_image))
code.interact(local=dict(globals(), **locals()))
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
    code.interact(local=dict(globals(), **locals()))

plt.imshow(a_channel_images[0,:,:,:])
plt.show()
# Load H - Channel
