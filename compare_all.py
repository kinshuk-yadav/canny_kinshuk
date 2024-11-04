import os
import cv2
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse
from scipy.ndimage import median_filter
from time import time
import random
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

from improved_canny_try import ImprovedCannyEdgeDetector
import numpy as np
from scipy.ndimage import convolve, gaussian_filter

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


def improved_canny_edge_detection(image, low_threshold=0.05, high_threshold=0.15, sigma=1.5):
    # Step 1: Gaussian filter to smooth the image (try a higher sigma for more smoothing)
    smoothed_image = gaussian_filter(image, sigma=sigma)
    
    # Step 2: Calculate gradient magnitude and direction (using Sobel operators)
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
    gradient_x = convolve(smoothed_image, sobel_x)
    gradient_y = convolve(smoothed_image, sobel_y)
    
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    # Non-maximum suppression as before
    suppressed = np.zeros_like(gradient_magnitude)
    angle = gradient_direction * (180.0 / np.pi)
    angle[angle < 0] += 180

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            try:
                q, r = 255, 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    suppressed[i, j] = gradient_magnitude[i, j]
                else:
                    suppressed[i, j] = 0
            except IndexError as e:
                pass

    # Double thresholding
    high_threshold_value = suppressed.max() * high_threshold
    low_threshold_value = high_threshold_value * low_threshold
    
    strong_edges = (suppressed >= high_threshold_value).astype(np.uint8)
    weak_edges = ((suppressed >= low_threshold_value) & (suppressed < high_threshold_value)).astype(np.uint8)

    # Edge tracking by hysteresis
    edges = np.zeros_like(suppressed, dtype=np.uint8)
    strong_i, strong_j = np.where(strong_edges == 1)
    edges[strong_i, strong_j] = 255

    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if weak_edges[i, j] == 1:
                if ((edges[i + 1, j - 1] == 255) or (edges[i + 1, j] == 255) or (edges[i + 1, j + 1] == 255)
                        or (edges[i, j - 1] == 255) or (edges[i, j + 1] == 255)
                        or (edges[i - 1, j - 1] == 255) or (edges[i - 1, j] == 255) or (edges[i - 1, j + 1] == 255)):
                    edges[i, j] = 255

    return edges / 255.0

def compute_metrics(original, filtered):
    mse_value = mse(original, filtered)
    psnr_value = psnr(original, filtered)
    return mse_value, psnr_value

def apply_custom_filter(image, kernel):
    """Apply a custom filter using convolution with a given kernel."""
    return cv2.filter2D(image, -1, kernel)

def to_binary(image, threshold=0.5):
    """Convert an image to binary (0 or 1) based on a threshold."""
    binary_img = np.where(image >= threshold, 1, 0)
    return binary_img

def apply_filters(image,noisy_image):
    results = {}
    results_imgs = []
    
    # Convert image to grayscale if it's colored
    if len(image.shape) == 3:
        image = rgb2gray(image)

    # Define filter kernels
    roberts_kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    log_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    # Apply Roberts filter
    start_time = time()
    roberts_img = np.sqrt(
        apply_custom_filter(noisy_image, roberts_kernel_x)**2 +
        apply_custom_filter(noisy_image, roberts_kernel_y)**2
    )
    roberts_binary = to_binary(roberts_img)
    roberts_mse, roberts_psnr = compute_metrics(image, roberts_binary)
    roberts_time = time() - start_time
    results_imgs.append(roberts_binary)
    
    results["Roberts"] = (roberts_mse, roberts_psnr, roberts_time)

    # Apply Prewitt filter
    start_time = time()
    prewitt_img = np.sqrt(
        apply_custom_filter(noisy_image, prewitt_kernel_x)**2 +
        apply_custom_filter(noisy_image, prewitt_kernel_y)**2
    )
    prewitt_binary = to_binary(prewitt_img)
    prewitt_mse, prewitt_psnr = compute_metrics(image, prewitt_binary)
    prewitt_time = time() - start_time
    results["Prewitt"] = (prewitt_mse, prewitt_psnr, prewitt_time)
    results_imgs.append(prewitt_binary)

    # Apply Sobel filter
    start_time = time()
    sobel_img = np.sqrt(
        apply_custom_filter(noisy_image, sobel_kernel_x)**2 +
        apply_custom_filter(noisy_image, sobel_kernel_y)**2
    )
    sobel_binary = to_binary(sobel_img)
    sobel_mse, sobel_psnr = compute_metrics(image, sobel_binary)
    sobel_time = time() - start_time
    results["Sobel"] = (sobel_mse, sobel_psnr, sobel_time)
    results_imgs.append(sobel_binary)

    # Apply Canny filter with preprocessing
    start_time = time()
    denoised_image = median_filter(noisy_image, size=3)  # Denoise
    # canny_img = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
    # canny_img = canny_img / 255  # Normalize

    # edge_detector = ImprovedCannyEdgeDetector([image])
    # edges_canny = edge_detector.detect()
    
    canny_img = improved_canny_edge_detection(denoised_image)
    canny_mse, canny_psnr = compute_metrics(image, canny_img)
    canny_time = time() - start_time
    results["Canny"] = (canny_mse, canny_psnr, canny_time)
    results_imgs.append(canny_img)

    # Apply LoG filter
    start_time = time()
    log_img = apply_custom_filter(image, log_kernel)
    log_binary = to_binary(log_img)
    log_mse, log_psnr = compute_metrics(image, log_binary)
    log_time = time() - start_time
    results["LoG"] = (log_mse, log_psnr, log_time)
    results_imgs.append(log_binary)

    # results_imgs = [roberts_img,sobel_img,prewitt_img,canny_img,log_img]
    return [results,results_imgs]

def visualize(imgs, format=None, gray=False):
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 3, plt_idx)
        plt.imshow(img, format)
    plt.show()

def process_directory(dir_name):
    data = []
    for filename in os.listdir(dir_name):
        file_path = os.path.join(dir_name, filename)
        if os.path.isfile(file_path):
            # Load and normalize the image
            image = imread(file_path, as_gray=True)

            # Apply filters and compute metrics
            noisy_image = add_noise(image)
            [results,results_imgs] = apply_filters(image, noisy_image)
            
            for operator, (mse_value, psnr_value, exec_time) in results.items():
                data.append({
                    "Image": os.path.splitext(filename)[0],
                    "Operator": operator,
                    "MSE": mse_value,
                    "PSNR": psnr_value,
                    "ET(s)": exec_time
                })
            data.append({})

            visualize(results_imgs)

    # Display data in tabular form
    df = pd.DataFrame(data)
    print(df)

# Example usage
dir_name = 'faces_imgs/processed'  # Replace with your directory path
process_directory(dir_name)

#########################################