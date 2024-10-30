import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Apply median filter to reduce salt & pepper noise
def apply_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# Apply wiener filter to reduce blur
def apply_wiener_filter(image):
    return wiener(image)

# Canny edge detection (ensure the image is 8-bit and grayscale)
def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    # Ensure image is 8-bit single-channel grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.uint8(image)  # Convert to 8-bit
    return cv2.Canny(image, low_threshold, high_threshold)

# Load image
def load_image(path, grayscale=True):
    if grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        return cv2.imread(path)

# Display multiple images together
def display_images(images, titles):
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(15, 5))
    for i in range(n):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def apply_sobel_edge_detection(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel on X-axis
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel on Y-axis
    sobel = cv2.magnitude(sobelx, sobely)  # Combine the two gradients
    return np.uint8(sobel)

def apply_prewitt_edge_detection(image):
    prewitt_kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Horizontal
    prewitt_kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])  # Vertical
    prewittx = cv2.filter2D(image, -1, prewitt_kernelx)
    prewitty = cv2.filter2D(image, -1, prewitt_kernely)
    prewitt = cv2.magnitude(prewittx, prewitty)
    return np.uint8(prewitt)

def apply_roberts_edge_detection(image):
    roberts_kernelx = np.array([[1, 0], [0, -1]])  # Horizontal
    roberts_kernely = np.array([[0, 1], [-1, 0]])  # Vertical
    robertsx = cv2.filter2D(image, -1, roberts_kernelx)
    robertsy = cv2.filter2D(image, -1, roberts_kernely)
    roberts = cv2.magnitude(robertsx, robertsy)
    return np.uint8(roberts)

def calculate_mse(original_image, edge_detected_image):
    return np.mean((original_image - edge_detected_image) ** 2)

def calculate_psnr(original_image, edge_detected_image):
    mse = calculate_mse(original_image, edge_detected_image)
    if mse == 0:  # No error
        return float('inf')
    max_pixel_value = 255.0
    return 20 * np.log10(max_pixel_value / np.sqrt(mse))


def main():
    # Load a noisy and blurred image
    image_path = 'lenna.jpeg'
    image = load_image(image_path)

    # Apply Median filter
    median_filtered = apply_median_filter(image, kernel_size=3)

    # Apply Wiener filter
    wiener_filtered = apply_wiener_filter(median_filtered)

    # Apply Edge Detection Algorithms
    regular_canny_edges = apply_canny_edge_detection(image)
    canny_edges = apply_canny_edge_detection(wiener_filtered)
    sobel_edges = apply_sobel_edge_detection(image)
    prewitt_edges = apply_prewitt_edge_detection(wiener_filtered)
    roberts_edges = apply_roberts_edge_detection(wiener_filtered)

    # Calculate PSNR for each method
    canny_psnr = calculate_psnr(image, canny_edges)
    sobel_psnr = calculate_psnr(image, sobel_edges)
    prewitt_psnr = calculate_psnr(image, prewitt_edges)
    roberts_psnr = calculate_psnr(image, roberts_edges)

    print(f"Canny PSNR: {canny_psnr}")
    print(f"Sobel PSNR: {sobel_psnr}")
    print(f"Prewitt PSNR: {prewitt_psnr}")
    print(f"Roberts PSNR: {roberts_psnr}")

    # Display all edge detection results together
    images = [image,regular_canny_edges, canny_edges, sobel_edges, prewitt_edges, roberts_edges]
    titles = ['Original Image','Regular Canny', 'Canny Edge Detection', 'Sobel Edge Detection', 'Prewitt Edge Detection', 'Roberts Edge Detection']
    
    display_images(images, titles)

if __name__ == '__main__':
    main()
