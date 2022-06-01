import eda.eda as eda

if __name__ == "__main__":

    TEST_IMAGE = "dataset/test/paper/testpaper01-00.png"
    RED = 0
    GREEN = 1
    BLUE = 2

    eda.create_rgb_channels_histogram(TEST_IMAGE)
    eda.create_basic_histogram(TEST_IMAGE,BLUE)