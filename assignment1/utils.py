import numpy as np
from PIL import Image
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
def clip(img, low=0, high=1):
    #clip image to display range
    return np.minimum(high, np.maximum(low, img))

def imread(filename):
    # Reads an RGB image and converts it into a floating point numpy array with
    # values between 0 and 1
    img = np.array(Image.open(filename).convert('RGB'))
    img = img.astype(np.float)/255.
    return img

def imwrite(img, filename):
    # Writes an image to file. Assumes image is either a floating point numpy array
    # with values between 0 and 1
    # or a uint8 array with values between 0 and 255
    if img.dtype!=np.uint8:
        img = (img*255).astype(np.uint8)

    img = Image.fromarray(img).save(filename)

def imshow(img):
    # wrapper around matplotlib's imshow function so that it produces more
    # intuitive behavior
    if len(img.shape)==3 and img.shape[2]==3:
        # RGB image; matplotlib does the right thing
        plt.imshow(img)
    else:
        # for graysccale images use a gray color scheme and stop matplotlib from
        # changing the dynamic range
        if img.dtype==np.uint8:
            plt.imshow(img, cmap=plt.get_cmap('gray'),vmin=0,vmax=255)
        else:
            plt.imshow(img, cmap=plt.get_cmap('gray'),vmin=0, vmax=1)
