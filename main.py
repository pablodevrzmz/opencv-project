import eda.eda as eda
import image_processing.processing as prc
import model.modelCreation as mdl
import glob
import os


#This method is pretty ugly
def loadDatasets():
    datasetChunk = ['test','train','validation']
    types = ['rock', 'paper', 'scissors']
    processedDatasets = ['datasets/Outline', 'datasets/InvertedColors', 'datasets/Contour']

    unprocessedDataset= 'datasets/Unprocessed'
    ##NOTE: This is creating a unwanted folder inside the validation directories.
    for i in datasetChunk:
        for j in types:
            subdir='/'+i+'/'+j+'/' if (i != 'validation') else '/'+i+'/'
            unprocessedImages = glob.glob(unprocessedDataset+subdir+'*.png')
            for k in unprocessedImages:
                if(not os.path.exists(k.replace('Unprocessed', 'Contour'))):
                    if (not os.path.exists('datasets/Contour/'+i+'/'+j)):
                        os.makedirs('datasets/Contour/'+i+'/'+j)
                    imageContour = prc.countourImage(k)
                    prc.saveImage(k.replace('Unprocessed', 'Contour'), imageContour)
                if(not os.path.exists(k.replace('Unprocessed', 'Outline'))):
                    if (not os.path.exists('datasets/Outline/'+i+'/'+j)):
                        os.makedirs('datasets/Outline/'+i+'/'+j)
                    imageOutline = prc.outlineImage(k)
                    prc.saveImage(k.replace('Unprocessed', 'Outline'), imageOutline)
                if(not os.path.exists(k.replace('Unprocessed', 'InvertedColors'))):
                    if (not os.path.exists('datasets/InvertedColors/'+i+'/'+j)):
                        os.makedirs('datasets/InvertedColors/'+i+'/'+j)
                    imageInverted = prc.invertImageColors(k) 
                    prc.saveImage(k.replace('Unprocessed', 'InvertedColors'), imageInverted)
                if(not os.path.exists(k.replace('Unprocessed', 'Blurred_Outline'))):
                    if (not os.path.exists('datasets/Blurred_Outline/'+i+'/'+j)):
                        os.makedirs('datasets/Blurred_Outline/'+i+'/'+j)
                    imageBlur = prc.blurredOutlineImage(k) 
                    prc.saveImage(k.replace('Unprocessed', 'Blurred_Outline'), imageBlur)
                if(not os.path.exists(k.replace('Unprocessed', 'Dilated_Outline'))):
                    if (not os.path.exists('datasets/Dilated_Outline/'+i+'/'+j)):
                        os.makedirs('datasets/Dilated_Outline/'+i+'/'+j)
                    imageDilated = prc.dilatedOutlineImage(k) 
                    prc.saveImage(k.replace('Unprocessed', 'Dilated_Outline'), imageDilated)

if __name__ == "__main__":

    TEST_IMAGE_1 = "datasets/Unprocessed/test/paper/testpaper01-00.png"
    TEST_IMAGE_2 = "datasets/Unprocessed/test/paper/testpaper03-15.png"
    RED = 0
    GREEN = 1
    BLUE = 2

    
    ### EDA

    #eda.create_rgb_channels_histogram(TEST_IMAGE_1)
    #eda.create_basic_histogram(TEST_IMAGE_1,BLUE)
    #distances =eda.compare_images_histograms(TEST_IMAGE_1,TEST_IMAGE_2)
    #print(distances)

    ### Transformations

    ## Thresholding and what else?
    #prc.runAllProccesingTypes(TEST_IMAGE_1)

    loadDatasets()
    mdl.runModels()
    ## Morphology
    #prc.apply_morphology(TEST_IMAGE_1,"EROSION")
    #prc.apply_morphology(TEST_IMAGE_1,"DILATION")

    ## Blurring
    #prc.apply_blur(TEST_IMAGE_1,"2D_CONVOLUTION")
    #prc.apply_blur(TEST_IMAGE_1,"BLUR")
    #prc.apply_blur(TEST_IMAGE_1,"GAUS_BLUR")
    #prc.apply_blur(TEST_IMAGE_1,"MEDIAN_BLUR")
    #prc.apply_blur(TEST_IMAGE_1,"BILATERAL")
    
