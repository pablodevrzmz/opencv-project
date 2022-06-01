# Tutorials https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
# Compare based on histograms: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def compare_images_histograms(image_path_1,image_path_2):
    pass

def create_basic_histogram(image_path,channel):
    img = cv.imread(image_path,channel)
    plt.hist(img.ravel(),256,[0,256])
    plt.title("Basic color histogram")
    plt.show()

def create_rgb_channels_histogram(image_path, normalize=True):
    
    src = cv.imread(image_path)

    color = ('b','g','r')

    for i,col in enumerate(color):
        histr = cv.calcHist([src],[i],None,[256],[0,256])
        if normalize:
                histr = cv.normalize(histr, histr, alpha=0, beta=400, norm_type=cv.NORM_MINMAX)
        plt.plot(histr,color = col)
        plt.xlim([0,256])

    plt.title("RGB Plot")
    plt.show()