from http.client import PRECONDITION_REQUIRED
import numpy as np
import pandas as pd
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121 as DN
from tensorflow.keras import Sequential
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, BatchNormalization
from keras.utils.vis_utils import plot_model

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


# Model architecture taken from: https://www.kaggle.com/code/kamalkhumar/multiclass-classification-with-image-augmentation
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
    
    print('Correctly predicted images from the test set: '+str(correctly_predicted_images))
    print('Wrongfully predicted images from test set: '+ str(len(y_test)-correctly_predicted_images))
    print('Accuracy: ' + str(correctly_predicted_images/len(y_test)))
    
    stats = []
    stats.append(metrics.roc_auc_score(predictions, y_test, average='macro'))
    stats.append(metrics.accuracy_score(predictions, y_test))
    stats.append(metrics.f1_score(predictions, y_test, average='macro'))
    stats.append(metrics.recall_score(predictions, y_test, average='macro'))
    stats.append(metrics.precision_score(predictions, y_test, average='macro'))
    return stats


#This method was done based on an answer from here: https://stackoverflow.com/questions/66635552/keras-assessing-the-roc-auc-of-multiclass-cnn
def getROCCurve(y, predictions, fig_title):
    target=['rock','paper','scissors']
    _, c_ax = plt.subplots(1,1, figsize = (12, 8))
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
    c_ax.set_title(fig_title)
    plt.savefig('Extracted_Data/'+fig_title+'.png')


def runModels():
    processing_methods=['Unprocessed','InvertedColors','Outline','Contour', 'Blurred_Outline', 'Dilated_Outline']
    models_ROC=[]
    models_accuracy=[]
    models_f1=[]
    models_recall=[]
    models_precision=[]
    index=0
    #Creating/Running each model.
    for i in range(0, len(processing_methods)):
        print("Processing Model " + processing_methods[i])
        model_name='saved_models/' + processing_methods[i]+str(index)
        model=''
        if os.path.isdir(model_name):
            model = load_model( model_name)
        else:
            model = createModel(processingMethod=processing_methods[i], model_index=index)
        
        X_test, y_test = loadDataset(processingMethod=processing_methods[i], dataChunkToLoad='test')
        model_stats=evaluateModel(model, X_test, y_test, processing_methods[i])
        models_ROC.append(model_stats[0])
        models_accuracy.append(model_stats[1])
        models_f1.append(model_stats[2])
        models_recall.append(model_stats[3])
        models_precision.append(model_stats[4])
    plot_model(model, to_file='Extracted_Data/model_architecture.png', show_shapes=True)
    #Creating the dataframe to store models' metrics
    statistics = pd.DataFrame(processing_methods, columns=['Processing_Method'])
    statistics['ROC'] = models_ROC
    statistics['Accuracy'] = models_accuracy
    statistics['F1 Score'] = models_f1
    statistics['Recall'] = models_recall
    statistics['Precision'] = models_precision
    statistics.to_csv('Extracted_Data/models_scores.csv', index=False)
