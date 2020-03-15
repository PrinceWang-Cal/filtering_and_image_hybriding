import matplotlib.pyplot as plt
from align_image_code import align_images
import image_operation as imop
import cv2
import skimage as sk
import skimage.io as skio
import numpy as np

# First load images

# high sf
im1 = plt.imread('./DerekPicture.jpg')/255.

# low sf
im2 = plt.imread('./nutmeg.jpg')/255

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im2, im1)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

def hybrid_image(im1, im2, sigma1, sigma2):

    """
    im1 is the low-freq image
    im2 is the high-freq image
    get high frequency

    :return:
    """
    im1 = sk.color.rgb2gray(im1)
    im2 = sk.color.rgb2gray(im2)

    low_freq = imop.gaussian_filter(im2, 6*sigma1, sigma1)
    skio.imsave("low.jpg", low_freq)
    im1_blur = imop.gaussian_filter(im1, 6*sigma2, sigma2)
    high_freq = im1 - im1_blur
    skio.imsave("high.jpg", high_freq)

    return low_freq + high_freq

sigma1 = 4
sigma2 = 9
hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
print(np.min(hybrid))
clipped_hybrid = np.clip(hybrid, -1, 1)
skio.imsave("merge.jpg", clipped_hybrid)

# for x in np.arange(1, 10):
#     for y in np.arange(1, 10):
#         hybrid = hybrid_image(im1_aligned, im2_aligned, x, y)
#         clipped_hybrid = np.clip(hybrid, -1, 1)
#         skio.imsave("hybrid_test" + "[" + str(x) + "," + str(y) + "].jpg", clipped_hybrid)



plt.imshow(hybrid)
plt.show

## Compute and display Gaussian and Laplacian Pyramids
## You also need to supply this function
N = 5 # suggested number of pyramid levels (your choice)
#pyramids(hybrid, N)








