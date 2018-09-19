import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools
import code
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from itertools import cycle
import math
import numpy as Math
from numpy import genfromtxt

def plotHistogram(x,xlabel,ylabel,title):
    plt.hist(x, normed=False, bins=10)
    mean = round(np.mean(x))
    min = np.min(x)
    max = np.max(x)
    std = np.round(np.std(x),3)
    plt.text(2000,50, "Mean="+str(mean) + "\n" + "Min="+str(min) + "\n" + "Max="+str(max) + "\n" + "Std="+str(std))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()




def plot2d(X, y, title):
	fig = plt.figure(figsize=(6,6))
	fig.suptitle(title, fontsize=20)
	ax = fig.add_subplot(111)
	ax.set_xlabel("n1",fontsize=12)
	ax.set_ylabel("n2",fontsize=12)
	ax.grid(True,linestyle='-',color='0.75')
	colors = ['r','g','b','black','white','c','m','y','#CCB974','#77BEDB']

	for i in np.unique(y):
		ax.scatter(X[np.where([y==i])[1], 0], X[np.where([y==i])[1], 1], c=colors[i-1], marker='o')

	ax.set_xlabel('n1')
	ax.set_ylabel('n2')

	plt.show()


def load_csv(file_path, delimiter):
    X = genfromtxt(file_path, delimiter=delimiter)
    return X

def round_keep_sum(cm, decimals=2):
	rcm = np.round(cm, decimals)
	for i in range(rcm.shape[0]):
		column = rcm[i,:]
		error = 1 - np.sum(column)
		sr = 10**(-decimals)
		n = int(round(error / sr))
		for _,j in sorted(((cm[i,j] - rcm[i,j], j) for j in range(cm.shape[1])), reverse=n>0)[:abs(n)]:
			rcm[i,j] += math.copysign(0.01, n)
	return rcm

# def plotNiceConfusionMatrix(y_test, y_pred, class_names):
# 	# Compute confusion matrix
# 	cnf_matrix = confusion_matrix(y_test, y_pred)
# 	np.set_printoptions(precision=2)
#
# 	# Plot non-normalized confusion matrix
# 	fig = plt.figure()
# 	conf_M(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
# 	fig.set_tight_layout(True)
#
# 	plt.show()

def plotBothConfusionMatrices(y_test, y_pred, class_names):
	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test, y_pred)
	# Plot non-normalized confusion matrix
	fig = plt.figure()
	conf_M2(cnf_matrix, classes=class_names, title='Confusion matrix')
	fig.set_tight_layout(True)
	# plt.show()
	return fig






# This function loads the data into X and y,
# outputs the feature names, the label dictionary, m and n

def loadnumpy(filename):
	array = np.load(filename)
	return array

def addOffset(X):
	X = np.c_[np.ones((X.shape[0],1)), X]
	return X

def normalize(DATA):
	for i in range(DATA.shape[1]):
		DATA[:,i] = (DATA[:,i] - np.mean(DATA[:,i]))/(np.max(DATA[:,i])-np.min(DATA[:,i])+.001)
	return DATA

def rms(y_true, y_pred):
	return mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')

# def plotConfusionMatrix(y, pred):
# 	n_labels = len(np.unique(y))
# 	matrix = np.zeros([n_labels+1,n_labels+1])
# 	matrix[0,1:] = np.unique(y)
# 	matrix[1:,0] = np.unique(y)
# 	matrix[1:,1:] = confusion_matrix(y.astype(int), pred.astype(int))
# 	print(matrix)


def round_keep_sum(cm, decimals=2):
	rcm = np.round(cm, decimals)
	for i in range(rcm.shape[0]):
		column = rcm[i,:]
		error = 1 - np.sum(column)
		sr = 10**(-decimals)
		n = int(round(error / sr))
		for _,j in sorted(((cm[i,j] - rcm[i,j], j) for j in range(cm.shape[1])), reverse=n>0)[:abs(n)]:
			rcm[i,j] += math.copysign(0.01, n)
	return rcm

def conf_M(cm):
    cmap=plt.cm.Blues
    classes = ["mytotic", "healthy"]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print('Confusion matrix')
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
