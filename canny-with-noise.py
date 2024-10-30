import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener

# Apply median filter to reduce salt & pepper noise
def apply_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# Apply Wiener filter to reduce blur
def apply_wiener_filter(image):
    return wiener(image)

# Canny edge detection (ensure the image is 8-bit and grayscale)
def apply_canny_edge_detection(image, low_threshold=50, high_threshold=150):
    # Ensure image is 8-bit single-channel grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.uint8(image)  # Convert to 8-bit
    return cv2.Canny(image, low_threshold, high_threshold)

# Add different types of noise
def add_salt_and_pepper_noise(image, noise_ratio=0.05):
    noisy_image = image.copy()
    total_pixels = noisy_image.size
    num_salt = np.ceil(noise_ratio * total_pixels * 0.5)
    num_pepper = np.ceil(noise_ratio * total_pixels * 0.5)

    # Salt noise (white pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in noisy_image.shape]
    noisy_image[tuple(coords)] = 255

    # Pepper noise (black pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in noisy_image.shape]
    noisy_image[tuple(coords)] = 0

    return noisy_image

# Gaussian blur
def add_gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Average blur
def add_average_blur(image, kernel_size=5):
    return cv2.blur(image, (kernel_size, kernel_size))

# Motion blur
def add_motion_blur(image, size=15):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    return cv2.filter2D(image, -1, kernel_motion_blur)

# Display multiple images in a grid format
def display_images_in_grid(images, titles, rows, cols):
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
    axs = axs.ravel()
    for i in range(len(images)):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Load original image (grayscale)
    image_path = 'lenna.jpeg'
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was successfully loaded
    if original_image is None:
        print(f"Error: Could not load image from path '{image_path}'. Please check the path.")
        return

    # Generate noisy and blurred images
    sp_noise_10 = add_salt_and_pepper_noise(original_image, 0.10)
    gaussian_blur = add_gaussian_blur(original_image)
    sp_noise_5 = add_salt_and_pepper_noise(original_image, 0.05)
    average_blur = add_average_blur(original_image)
    motion_blur = add_motion_blur(original_image)

    # Process each noisy/blurred image with filtering and Canny edge detection
    images = []
    titles = []

    # Noisy/blurred images
    noisy_images = [sp_noise_10, gaussian_blur, sp_noise_5, average_blur, motion_blur]
    noisy_titles = ['Simple Blur + 10% Salt&Pepper', 'Gaussian Blur', '5% Salt&Pepper',
                    'Average Blur', 'Motion Blur']

    # Append original images
    images.append(original_image)
    titles.append("Original Image")

    # Append noisy images
    images.extend(noisy_images)
    titles.extend(noisy_titles)

    # Apply filters and edge detection to each noisy image
    for noisy_image in noisy_images:
        # Apply Wiener and Median filters
        wiener_filtered = apply_wiener_filter(noisy_image)
        median_filtered = apply_median_filter(noisy_image)

        # Edge detection
        canny_edges_wiener = apply_canny_edge_detection(wiener_filtered)
        canny_edges_median = apply_canny_edge_detection(median_filtered)

        # Add results to images list for display
        images.append(canny_edges_wiener)
        images.append(canny_edges_median)

        # Titles for edge-detected images
        titles.append(f"Canny on Wiener Filtered")
        titles.append(f"Canny on Median Filtered")

    # Display images in a grid
    display_images_in_grid(images, titles, 4, 4)  # Adjust rows/cols as per images

if __name__ == '__main__':
    main()
