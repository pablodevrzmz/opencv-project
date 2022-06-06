import eda.eda as eda
import image_processing.processing as prc

if __name__ == "__main__":

    TEST_IMAGE_1 = "dataset/test/paper/testpaper01-00.png"
    TEST_IMAGE_2 = "dataset/test/paper/testpaper03-15.png"

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

    ## Morphology
    #prc.apply_morphology(TEST_IMAGE_1,"EROSION")
    #prc.apply_morphology(TEST_IMAGE_1,"DILATION")
