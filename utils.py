import numpy as np
# import skimage
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import os
import scipy.misc as sm

# from skimage.color import rgb2gray
import random

def add_noise(img, noise_type="salt_and_pepper", percentage_of_noise=0.2):
    if noise_type == "salt_and_pepper":
        noisy_img = img.copy()
        total_pixels = img.size
        num_noisy_pixels = int(percentage_of_noise * total_pixels)

        # Add salt noise
        for _ in range(num_noisy_pixels // 2):
            x = random.randint(0, img.shape[0] - 1)
            y = random.randint(0, img.shape[1] - 1)
            noisy_img[x, y] = 1  # White pixel

        # Add pepper noise
        for _ in range(num_noisy_pixels // 2):
            x = random.randint(0, img.shape[0] - 1)
            y = random.randint(0, img.shape[1] - 1)
            noisy_img[x, y] = 0  # Black pixel

    elif noise_type == "gaussian":
        mean = 0
        std_dev = 0.1 * percentage_of_noise
        gaussian_noise = np.random.normal(mean, std_dev, img.shape)
        noisy_img = img + gaussian_noise
        noisy_img = np.clip(noisy_img, 0, 1)  # Keep pixel values in the range [0, 1]

    else:
        raise ValueError("Unsupported noise type. Choose 'salt_and_pepper' or 'gaussian'.")
    
    return noisy_img

def load_with_noise(dir_name='faces_imgs', noise_type="salt_and_pepper", percentage_of_noise=0.2):
    """
    Load images from a specified directory, convert to grayscale, and apply noise.
    
    Parameters:
        dir_name (str): Directory containing image files.
        noise_type (str): Type of noise to apply ('salt_and_pepper' or 'gaussian').
        percentage_of_noise (float): Percentage of image pixels to apply noise to.
        
    Returns:
        List of noisy grayscale images.
    """
    imgs = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(os.path.join(dir_name, filename)):
            img = mpimg.imread(os.path.join(dir_name, filename))
            img = rgb2gray(img)
            img_noisy = add_noise(img, noise_type, percentage_of_noise)
            imgs.append(img_noisy)
    return imgs



def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def load_data(dir_name = 'canny_kinshuk\faces_imgs'):    
    '''
    Load images from the "faces_imgs" directory
    Images are in JPG and we convert it to gray scale images
    '''
    imgs = []
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            img = mpimg.imread(dir_name + '/' + filename)
            img = rgb2gray(img)
            imgs.append(img)
    return imgs


def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()

def temp_visualize(imgs,format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        plt.imshow(img, format)
    plt.show()