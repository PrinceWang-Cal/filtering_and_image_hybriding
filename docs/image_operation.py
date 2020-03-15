import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr
import glob
from skimage.filters import roberts
import skimage.data as data
import scipy
import matplotlib.pyplot as plt
import cv2

# cut into three function
def split_im(image):
    height = np.floor(image.shape[0] / 3.0).astype(np.int)
    image1 = image[:height]
    image2 = image[height:2 * height]
    image3 = image[2 * height:3 * height]
    return image1, image2, image3


def show_im(im):
    skio.imshow(im)
    skio.show()


def crop_im(im, percentage):
    """
    crop the image in all four sides by <n> pixels
    """
    height = im.shape[0]
    width = im.shape[1]
    n = int(height * percentage)
    m = int(width * percentage)
    return im[n:-n, m:-m]


def crop_im_abs(im, dim):
    h = dim[0]
    w = dim[1]
    return im[h:-h, w:-w]


def resize_im(im, factor):
    size = (np.array(im.shape) * factor).astype(int)
    res_im = sktr.resize(im, size, order=3)
    return res_im


def shift_im(im, right, down):
    im = np.roll(im, right, axis=1)
    im = np.roll(im, down, axis=0)
    return im


def read_im(im):
    im = skio.imread(im)
    return im

def stack_channel(R, G, B):
    im_stacked = np.stack(np.asarray([B, G, R]), axis = 2)
    return im_stacked


def dy_filter(im):
    """
    compute the Dy of the image

    """

    Dy = np.array([[1], [-1]])
    filtered = scipy.signal.convolve2d(im, Dy, mode="same")
    return normalize(filtered)


def dx_filter(im):
    """
    compute the Dx of the image

    """
    Dx = np.array([[1, -1]])
    filtered = scipy.signal.convolve2d(im, Dx, mode="same")
    return normalize(filtered)


def normalize(im):
    return im / (np.max(im) - np.min(im))

def compute_grad_mag(Dx, Dy):
    grad_mag = np.sqrt(Dx**2 + Dy**2)
    return normalize(grad_mag)


def grad_to_edge(threshold=0.10, grad_im=None):
    """
    binarizes the gradient magnitude image
    """
    boolean_array = grad_im > threshold
    zero_one_array = boolean_array.astype(np.int)

    return zero_one_array

def generate_gaussian_filter(ksize, sigma):
    kernel1d = cv2.getGaussianKernel(ksize, sigma = sigma)
    kernel1d_T = kernel1d_T = np.transpose(kernel1d)
    return np.outer(kernel1d, kernel1d_T)

def gaussian_filter(im, ksize, sigma):
    gaussian = generate_gaussian_filter(ksize, sigma)
    gaus_im = scipy.signal.convolve2d(im, gaussian, mode = "same")
    return gaus_im


# ToDO: make this function more efficient
def filter_im_gaussian(im):
    guassian_filter = generate_gaussian_filter(12, 2)
    gaus_im = scipy.signal.convolve2d(im, gaussian_filter, mode="same")

    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])

    filtered_hori = scipy.signal.convolve2d(im, Dx, mode="same")
    filtered_vert = scipy.signal.convolve2d(im, Dy, mode="same")

    grad_im = compute_grad_mag(filtered_hori, filtered_vert)
    normalized_grad = grad_im / (np.max(grad_im) - np.min(grad_im))

    return grad_to_edge(normalized_grad)

def gaussian_filter_color(im, ksize, sigma):
    imb, img, imr = cv2.split(im)
    imb_blur = gaussian_filter(imb, ksize, sigma)
    img_blur = gaussian_filter(img, ksize, sigma)
    imr_blur = gaussian_filter(imr, ksize, sigma)
    return stack_channel(imr_blur, img_blur, imb_blur)

def compute_grad_angle(im):
    Dy = dy_filter(im)
    Dx = dx_filter(im)
    dy_over_dx = Dy / Dx
    grad_directions = np.arctan2(Dy, Dx)
    return grad_directions


def count_horizontal(im, epsilon=np.pi / 180):
    imf = im.flatten()
    result1 = imf < epsilon
    result2 = imf > -epsilon
    result12 = result1 & result2

    result3 = imf < np.pi + epsilon
    result4 = imf > np.pi - epsilon
    result34 = result3 & result4

    result = np.logical_or(result12, result34)
    return np.sum(result)


def count_vertical(im, epsilon=np.pi / 180):
    half_pi = np.pi / 2
    imf = im.flatten()
    result1 = imf < half_pi + epsilon
    result2 = imf > half_pi - epsilon
    result12 = result1 & result2

    result3 = imf < half_pi + epsilon
    result4 = imf > half_pi - epsilon
    result34 = result3 & result4

    result = np.logical_or(result12, result34)
    return np.sum(result)


def sharpen(im, alpha):
    """
    the image should be floats
    """
    imb, img, imr = cv2.split(im)

    imb_blur = gaussian_filter(imb, 5, 1)
    img_blur = gaussian_filter(img, 5, 1)
    imr_blur = gaussian_filter(imr, 5, 1)

    imb_high = imb - imb_blur
    img_high = img - img_blur
    imr_high = imr - imr_blur

    imb_sharpen = np.clip(imb + (alpha * imb_high), 0, 1)
    img_sharpen = np.clip(img + (alpha * img_high), 0, 1)
    imr_sharpen = np.clip(imr + (alpha * imr_high), 0, 1)

    im_sharpen = stack_channel(imr_sharpen, img_sharpen, imb_sharpen)
    return im_sharpen
