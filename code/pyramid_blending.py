

# Import required libraries
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

# Read source, target and mask for a given id
def Read(id, path = ""):
    source = plt.imread(path + "source_" + id + ".jpg")
    info = np.iinfo(source.dtype) # get information about the image type (min max values)
    source = source.astype(np.float32) / info.max # normalize the image into range 0 and 1
    target = plt.imread(path + "target_" + id + ".jpg")
    info = np.iinfo(target.dtype) # get information about the image type (min max values)
    target = target.astype(np.float32) / info.max # normalize the image into range 0 and 1
    mask   = plt.imread(path + "mask_" + id + ".jpg")
    info = np.iinfo(mask.dtype) # get information about the image type (min max values)
    mask = mask.astype(np.float32) / info.max # normalize the image into range 0 and 1

    return source, mask, target

def ensure_consistent_dimensions(source, target, mask):
    # Determine the maximum dimensions
    max_height = max(source.shape[0], target.shape[0], mask.shape[0])
    max_width = max(source.shape[1], target.shape[1], mask.shape[1])

    # Resize source, target, and mask to match the maximum dimensions
    source_resized = resize(source, (max_height, max_width), anti_aliasing=True)
    target_resized = resize(target, (max_height, max_width), anti_aliasing=True)
    mask_resized = resize(mask, (max_height, max_width), anti_aliasing=True)

    return source_resized, target_resized, mask_resized


def gaussian(img, lvl):
    pyramid = [img]
    for i in range(lvl):
        img = resize(img, (img.shape[0]//2, img.shape[1]//2), anti_aliasing=True)
        pyramid.append(img)
    return pyramid

def laplacian(img, lvl):
    gaussian_pyramid = gaussian(img, lvl)
    pyramid = []
    for i in range(lvl):
        next_img = gaussian_pyramid[i] - resize(gaussian_pyramid[i+1], gaussian_pyramid[i].shape, anti_aliasing=True)
        pyramid.append(next_img)
    pyramid.append(gaussian_pyramid[-1])
    return pyramid

def collapse(pyramid):
    img = pyramid[-1]
    for i in reversed(pyramid[:-1]):
        img = resize(img, i.shape, anti_aliasing=True) + i
    return img


# Pyramid Blend
def PyramidBlend(source, mask, target):

    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)
    if mask.shape[2] == 1:
        mask = np.repeat(mask, 3, axis=2)
    
    mask_pyramid = gaussian(mask, 5)
    source_pyramid = laplacian(source, 5)
    target_pyramid = laplacian(target, 5)

    blended_pyramid = []

    for i in range(len(mask_pyramid)):
        blended_layer = mask_pyramid[i] * source_pyramid[i] + (1 - mask_pyramid[i]) * target_pyramid[i]
        blended_pyramid.append(blended_layer)

    return source * mask + target * (1 - mask)



if __name__ == '__main__':
    # Setting up the input output paths
    inputDir = '../Images/'
    outputDir = '../Results/'

    # main area to specify files and display blended image

    index = 2

    # Read data and clean mask
    
    source, maskOriginal, target = Read(str(index).zfill(2), inputDir)

    source, target, maskOriginal = ensure_consistent_dimensions(source, target, maskOriginal)

    # Cleaning up the mask
    mask = np.ones_like(maskOriginal)
    mask[maskOriginal < 0.5] = 0

    
    ### The main part of the code ###

    # Implement the PyramidBlend function (Task 2)
    pyramidOutput = PyramidBlend(source, mask, target)

    

    
    # Writing the result
    np.clip(pyramidOutput, 0, 1, out=pyramidOutput)
    plt.imsave("{}pyramid_{}.jpg".format(outputDir, str(index).zfill(2)), pyramidOutput)
