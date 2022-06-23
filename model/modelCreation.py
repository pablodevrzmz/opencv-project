import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121 as DN
from tensorflow.keras import Sequential, metrics
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, BatchNormalization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Labels: [1,0,0]=rock, [0,1,0]=paper, [0,0,1]=scissors
def loadImages(path, label):
    images = glob.glob(path+'/*.png')
    images_data = []
    labels = []
    #Loading Images and creating label array
    for i in range(0, len(images)):
        labels.append(label)
        img = cv.imread(images[i])
        images_data.append(img/255)
    return images_data, np.array(labels)
        

def loadDataset(processingMethod, dataChunkToLoad):
    paths = []
    labels = [[1,0,0], [0,1,0], [0,0,1]]
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
        X_partial, y_partial = loadImages(paths[i], labels[i])
        X_len = len(X_partial) + X_len
        X = np.append(X, X_partial)
        y = np.append(y, y_partial)
    
    X = X.reshape(X_len, 300, 300, 3)
    y = y.reshape(X_len, 3)
    # Shuffling dataset, X and y items share the same index
    shuffledIndexes = np.random.permutation(len(y))
    X = X[shuffledIndexes]
    y = y[shuffledIndexes]

    return X, y


def createModel(processingMethod='Unprocessed', model_index=0):
    X_train, y_train = loadDataset(processingMethod, 'train')
    X_test, y_test = loadDataset(processingMethod, 'test')

    model_name=processingMethod+str(model_index)
    #Creating the DenseNet model
    #References: https://www.pluralsight.com/guides/introduction-to-densenet-with-tensorflow
    #            https://bouzouitina-hamdi.medium.com/transfer-learning-with-keras-using-densenet121-fffc6bb0c233
    denseArchitecture = DN(weights="imagenet", include_top=False, input_shape=(300, 300, 3))
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
    for layer in denseArchitecture.layers[:121]:
        layer.trainable = False #we are using imagenet, so this layers shouldn't be trainable

    #input = denseArchitecture.input
    #added_layers = MaxPooling2D()(denseArchitecture.output)
    #added_layers = BatchNormalization()(added_layers)
    #added_layers = Dense(512, activation='relu')(added_layers)
    #added_layers = Dropout(0.33)(added_layers)
    #added_layers = Flatten()(added_layers)
    #added_layers = Dense(64, activation='relu')(added_layers)
    #added_layers = Dense(3, activation='softmax')(added_layers)
    #model = Model(input, added_layers)
    model = Sequential() #DenseNet is a sequential architecture
    model.add(denseArchitecture)
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.33))
    #Activations: https://www.tensorflow.org/api_docs/python/tf/keras/activations
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(3, activation="softmax"))
    #Reference: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    model.compile(optimizer=Adam(learning_rate=0.001), metrics=['accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall')], loss='categorical_crossentropy')
    model_metrics = model.fit(x=X_train, y=y_train, epochs=10, validation_data=(X_test, y_test))
    model.save(model_name)
    model.evaluate(x=X_test, y=y_test)
    plt.plot(model_metrics.history['accuracy'], color='blue', label='train')
    plt.plot(model_metrics.history['val_accuracy'], color='orange', label='test')
    plt.show()
