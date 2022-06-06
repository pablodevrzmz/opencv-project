# Tutorials https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
# Compare based on histograms: https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html

import numpy as np
import cv2
from matplotlib import pyplot as plt

def compare_images_histograms(image_path_1,image_path_2, bins=8):
    
    DISTANCE_FUNCTIONS = (
        ("HISTCMP_CORREL",cv2.HISTCMP_CORREL),
        ("HISTCMP_CHISQR",cv2.HISTCMP_CHISQR),
        ("HISTCMP_INTERSECT",cv2.HISTCMP_INTERSECT),
        ("HISTCMP_BHATTACHARYYA",cv2.HISTCMP_BHATTACHARYYA),
    )

    # Prepare histrograms
    img_1 = cv2.imread(image_path_1)
    img_2 = cv2.imread(image_path_2)

    hist_1 = cv2.calcHist([img_1], [0, 1, 2], None, [bins, bins, bins],[0, 256, 0, 256, 0, 256])
    hist_1 = cv2.normalize(hist_1, hist_1).flatten()
    
    hist_2 = cv2.calcHist([img_2], [0, 1, 2], None, [bins, bins, bins],[0, 256, 0, 256, 0, 256])
    hist_2 = cv2.normalize(hist_2, hist_2).flatten()

    # Run comparisons
    distances = dict()

    for df in DISTANCE_FUNCTIONS:
        distances[df[0]] = cv2.compareHist(hist_1, hist_2, df[1])
    
    return distances


def create_basic_histogram(image_path,channel):
    img = cv2.imread(image_path,channel)
    plt.hist(img.ravel(),256,[0,256])
    plt.title("Basic color histogram")
    plt.show()

def create_rgb_channels_histogram(image_path, normalize=True):
    
    src = cv2.imread(image_path)

    color = ('b','g','r')

    for i,col in enumerate(color):
        histr = cv2.calcHist([src],[i],None,[256],[0,256])
        if normalize:
                histr = cv2.normalize(histr, histr, alpha=0, beta=400, norm_type=cv2.NORM_MINMAX)
        plt.plot(histr,color = col)
        plt.xlim([0,256])

    plt.title("RGB Plot")
    plt.show()