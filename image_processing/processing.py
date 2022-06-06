#Documentation: https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html


import numpy as np
import cv2 as cv
import numpy as np

def showImage(img, windowName):
    cv.imshow(windowName, img)


def invertImageColors(imagePath):
    img = cv.imread(imagePath)
    return cv.bitwise_not(img)

def upscaleImage(imagePath):
    #Couldn't find it, resize?
    print('do something')


#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#types of thresholding
#cv.THRESH_BINARY (default)
#cv.THRESH_BINARY_INV
#cv.THRESH_TRUNC
#cv.THRESH_TOZERO
#cv.THRESH_TOZERO_INV

###Adaptive methods
#cv.ADAPTIVE_THRESH_MEAN_C
#cv.ADAPTIVE_THRESH_GAUSSIAN_C

def thresholdImage(imagePath, color=None, thresholdValue=127, maxValue=255, threshMethod=cv.THRESH_BINARY, adaptativeThreshMethod=None, blockSize=None, C=None, useOtsu=False):
    if useOtsu:
        color=cv.IMREAD_GRAYSCALE
    
    img = cv.imread(imagePath, color)
    if(adaptativeThreshMethod != None): #color must be 0 for the adaptative threshold to work
        thresh = cv.adaptiveThreshold(img, maxValue, adaptativeThreshMethod, threshMethod+cv.THRESH_OTSU if useOtsu else threshMethod, blockSize, C)
    else:
        _, thresh = cv.threshold(img, thresholdValue, maxValue, threshMethod+cv.THRESH_OTSU if useOtsu else threshMethod)
    return thresh


def runAllProccesingTypes(imagePath):
    showImage(cv.imread(imagePath), "Original")
    showImage(invertImageColors(imagePath), "Inverted")
    showImage(thresholdImage(imagePath), "Binary Threshold")
    showImage(thresholdImage(imagePath, threshMethod=cv.THRESH_BINARY_INV, useOtsu=True), "Binary Inverted Otsu Threshold")#This gets the contour, though it requires the colors to be in a greyscale
    showImage(thresholdImage(imagePath, color=cv.IMREAD_GRAYSCALE, threshMethod=cv.THRESH_BINARY_INV), "Binary Inverted GreyScale Threshold") 
    showImage(thresholdImage(imagePath, threshMethod=cv.THRESH_BINARY_INV), "Binary Inverted Threshold") 
    showImage(thresholdImage(imagePath, threshMethod=cv.THRESH_TRUNC), "Truncated Threshold")
    showImage(thresholdImage(imagePath, threshMethod=cv.THRESH_TOZERO), "ToZero Threshold")
    showImage(thresholdImage(imagePath, threshMethod=cv.THRESH_TOZERO_INV), "ToZero Inverted Threshold")
    showImage(thresholdImage(imagePath, color=0 ,adaptativeThreshMethod=cv.ADAPTIVE_THRESH_MEAN_C, blockSize=11, C=2), "Adaptive Gaussian Threshold")
    cv.waitKey(0) #Press any key to close all images' windows
    cv.destroyAllWindows()


def apply_morphology(image_path,method, kernel=(np.ones((5,5), np.uint8)), iterations=1, show=True):

    img = cv.imread(image_path)

    if method == "EROSION":
        output_image = cv.cv2.erode(img, kernel, iterations)  
    elif method == "DILATION": 
        output_image = cv.cv2.dilate(img, kernel, iterations) 
    
    if show:
        showImage(output_image,method)
        cv.waitKey(0)
    
    return output_image
