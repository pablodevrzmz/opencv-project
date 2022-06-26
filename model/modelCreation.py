from http.client import PRECONDITION_REQUIRED
import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121 as DN
from tensorflow.keras import Sequential
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
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
    labels = [[1,0,0], [0,1,0], [0,0,1]] #[00, 010, 01]
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
    model_name=processingMethod+str(model_index)
    #Model Definition
    model = Sequential() #DenseNet is a sequential architecture
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(300,300,3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


    model_metrics = model.fit(x=X_train, y=y_train, epochs=10, validation_split=.3)
    model.save('saved_models/' + model_name)
    return model

def evaluateModel(model, X_test, y_test, title):
    predictions = model.predict(X_test)
    correctly_predicted_images=0
    for i in range(0, len(predictions)):
        #getting the predicted class []
        predictions[i][0] = 1 if predictions[i][0]>=0.5 else 0
        predictions[i][1] = 1 if predictions[i][1]>=0.5 else 0
        predictions[i][2] = 1 if predictions[i][2]>=0.5 else 0
        if(predictions[i][0] == y_test[i][0] and predictions[i][1] == y_test[i][1] and predictions[i][2] == y_test[i][2]):
            correctly_predicted_images+=1
    
    ROC = metrics.roc_auc_score(predictions, y_test)
    accuracy = metrics.accuracy_score(predictions, y_test)
    print('ROC score: ' + str(ROC))
    print('accuracy method score: ' + str(accuracy))
    print('Correctly predicted images from the test set: '+str(correctly_predicted_images))
    print('Wrongfully predicted images from test set: '+ str(len(y_test)-correctly_predicted_images))
    print('Accuracy: ' + str(correctly_predicted_images/len(y_test)))
    print('ROC AUC score:', getROCCurve(y_test, predictions))
    plt.title(title)
    plt.show()

target=['rock','paper','scissors']


def getROCCurve(y, predictions):
    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
    lb = LabelBinarizer()
    lb.fit(y)
    y_test = lb.transform(y)
    predictions = lb.transform(predictions)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = metrics.roc_curve(y_test[:,idx].astype(int), predictions[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, metrics.auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    return metrics.roc_auc_score(y_test, predictions, average='macro')

