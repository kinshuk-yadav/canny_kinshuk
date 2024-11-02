import cupy as cp
import cv2
import os
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import convolve as cp_convolve  # Import GPU-compatible convolution

class CannyEdgeDetectorGPU:
    def __init__(self, img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.img = cp.array(img, dtype=cp.float32)
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = cp.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * cp.pi * sigma**2)
        g = cp.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
        return g.astype(cp.float32)  # Ensure the kernel is a float32 array

    def sobel_filters(self):
        Kx = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
        Ky = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=cp.float32)
        Ix = cp_convolve(self.img, Kx, mode='reflect')
        Iy = cp_convolve(self.img, Ky, mode='reflect')

        G = cp.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = cp.arctan2(Iy, Ix)
        return G, theta

    def non_max_suppression(self, img, theta):
        M, N = img.shape
        Z = cp.zeros((M, N), dtype=cp.float32)
        angle = theta * 180. / cp.pi
        angle[angle < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

        return Z

    def threshold(self, img):
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        res = cp.zeros_like(img, dtype=cp.float32)
        weak = cp.float32(self.weak_pixel)
        strong = cp.float32(self.strong_pixel)

        strong_i, strong_j = cp.where(img >= highThreshold)
        zeros_i, zeros_j = cp.where(img < lowThreshold)

        weak_i, weak_j = cp.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res

    def hysteresis(self, img):
        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i, j] == weak):
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
        return img

    def detect(self):
        img_smoothed = cp_convolve(self.img, self.gaussian_kernel(self.kernel_size, self.sigma), mode='reflect')
        gradientMat, thetaMat = self.sobel_filters()
        nonMaxImg = self.non_max_suppression(gradientMat, thetaMat)
        thresholdImg = self.threshold(nonMaxImg)
        img_final = self.hysteresis(thresholdImg)
        return cp.asnumpy(img_final)

# Load image and convert to grayscale
img_path = "faces_imgs/sample_face.jpg"  # Replace with the path to your image file in faces_imgs
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Initialize the Canny detector and detect edges
canny_detector = CannyEdgeDetectorGPU(img, sigma=1.4, kernel_size=5, lowthreshold=0.05, highthreshold=0.15)
edges = canny_detector.detect()

# Plot original and edge-detected images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Edge Detected Image')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()
