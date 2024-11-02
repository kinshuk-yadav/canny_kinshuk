import cupy as cp
from cupyx.scipy.ndimage import median_filter, convolve  # Import median_filter from cupyx
from scipy.signal import wiener  # CuPy does not have wiener filter, so we use SciPy here

class ImprovedCannyEdgeDetectorGPU:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        # Convert input images to CuPy arrays for GPU processing
        self.imgs = [cp.array(img) for img in imgs]
        self.imgs_final = []
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
        return g

    def sobel_filters(self, img):
        Kx = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=cp.float32)
        Ky = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=cp.float32)
        
        Ix = cp.convolve(img, Kx, mode='constant')
        Iy = cp.convolve(img, Ky, mode='constant')
        
        G = cp.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = cp.arctan2(Iy, Ix)
        return (G, theta)

    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = cp.zeros((M, N), dtype=cp.int32)
        angle = D * 180. / cp.pi
        angle[angle < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
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
                except IndexError as e:
                    pass
        return Z

    def threshold(self, img):
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        res = cp.zeros((M, N), dtype=cp.int32)

        weak = cp.int32(self.weak_pixel)
        strong = cp.int32(self.strong_pixel)

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
                if img[i, j] == weak:
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img

    def detect(self):
        imgs_final = []
        for img in self.imgs:
            # Apply median filter from cupyx
            img = median_filter(img, size=3)
            # Apply Wiener filter (still using CPU, convert back to CuPy after)
            img = cp.array(wiener(cp.asnumpy(img)))  # Convert back to CuPy array after Wiener filtering

            # Use cupyx's convolve for 2D convolution
            self.img_smoothed = convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma), mode='constant')
            self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)
            self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self.threshold(self.nonMaxImg)
            img_final = self.hysteresis(self.thresholdImg)
            self.imgs_final.append(img_final)
        
        # Convert results back to NumPy arrays if needed for final output
        return [cp.asnumpy(img) for img in self.imgs_final]