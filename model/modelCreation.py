import numpy as np
import cv2 as cv
import glob
from tensorflow.keras.applications import DenseNet121 as DN
from tensorflow.keras import Sequential, metrics
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, BatchNormalization

#Labels: 0=rock, 1=paper, 2=scissors
def loadImages(path, label):
    images = glob.glob(path+'/*.png')
    images_data = []
    labels = []
    if label==1:
        labels = np.ones(len(images))
    else:
        labels = np.zeros(len(images))
    #Loading Images and creating label array
    for i in range(0, len(images)):
        if (label != 0 and label != 1):
            labels[i] = label
        img = cv.imread(images[i])
        images_data.append(img)
    return images_data, labels
        

def loadDataset(processingMethod, dataChunkToLoad):
    paths = []
    if dataChunkToLoad=='train':
        paths = ['./datasets/'+processingMethod+'/train/rock', './datasets/'+processingMethod+'/train/paper', './datasets/'+processingMethod+'/train/scissors']
    elif dataChunkToLoad=='test':
        paths = ['datasets/'+processingMethod+'/test/rock', 'datasets/'+processingMethod+'/test/paper', 'datasets/'+processingMethod+'/test/scissors']
    else: #This should be handled differently, or fix the validation data into folders too
        path_to_validation='datasets/'+processingMethod+'/validation'
    
    #Loading images for each label
    X = []
    y = []
    X_len = 0
    for i in range(0,len(paths)):
        X_partial, y_partial = loadImages(paths[i], i)
        X_len = len(X_partial) + X_len
        X = np.append(X, X_partial)
        y = np.append(y, y_partial)
    
    X = X.reshape(X_len, 300, 300, 3)
    # Shuffling dataset, X and y items share the same index
    shuffledIndexes = np.random.permutation(len(y))
    X = X[shuffledIndexes]
    y = y[shuffledIndexes]

    return X, y


def createModel(processingMethod='Unprocessed'):
    X_train, y_train = loadDataset(processingMethod, 'train')
    X_test, y_test = loadDataset(processingMethod, 'test')
    print(y_train.shape)
    print(X_train.shape)

    #This is where the model work start
