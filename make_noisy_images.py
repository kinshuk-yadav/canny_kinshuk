import cv2
import numpy as np
from scipy.ndimage import convolve
import os

# Path to input image and output directory
input_image_path = './faces_imgs/lenna.jpeg'
output_dir = './faces_imgs/processed/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the image
image = cv2.imread(input_image_path)
if image is None:
    print("Error: Could not load image.")
else:
    print("Image loaded successfully.")

# Function to apply motion blur
def apply_motion_blur(image, theta):
    size = theta
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    blurred = cv2.filter2D(image, -1, kernel_motion_blur)
    return blurred

# Function to add Gaussian noise
def add_gaussian_noise(image, sigma):
    noisy_image = image + sigma * np.random.normal(loc=0, scale=1, size=image.shape) * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Function to add Salt & Pepper noise
def add_salt_and_pepper_noise(image, percentage):
    noisy_image = np.copy(image)
    num_salt = np.ceil(percentage * image.size * 0.5).astype(int)
    num_pepper = np.ceil(percentage * image.size * 0.5).astype(int)
    
    # Salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255
    
    # Pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0
    
    return noisy_image

# Apply motion blur with different θ values
for theta in [10, 30]:
    blurred_image = apply_motion_blur(image, theta)
    output_path = os.path.join(output_dir, f'lenna_motion_blur_theta_{theta}.jpeg')
    cv2.imwrite(output_path, blurred_image)

# Apply Gaussian noise with different σ values
for sigma in [0.01, 0.03, 0.05, 0.1, 0.3]:
    noisy_image = add_gaussian_noise(image, sigma)
    output_path = os.path.join(output_dir, f'lenna_gaussian_noise_sigma_{sigma}.jpeg')
    cv2.imwrite(output_path, noisy_image)

# Apply Salt & Pepper noise with different percentages
for percentage in [0.05, 0.1, 0.2, 0.3]:
    noisy_image = add_salt_and_pepper_noise(image, percentage)
    output_path = os.path.join(output_dir, f'lenna_salt_pepper_noise_{int(percentage*100)}.jpeg')
    cv2.imwrite(output_path, noisy_image)

print("All degraded images have been saved in the processed folder.")
