import cupy as cp
from cupyx.scipy import ndimage
import time

class CannyEdgeDetectorCUDA:
    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        self.imgs = [cp.array(img) for img in imgs]
        self.imgs_final = []
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold

    def gaussian_kernel(self, size, sigma=1):
        # start_gaus = time.time()
        size = int(size) // 2
        x, y = cp.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * cp.pi * sigma**2)
        g = cp.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
        # end_gaus = time.time()
        # print(f"Time for gaussian kernel: {end_gaus - start_gaus} seconds")
        return g

    def sobel_filters(self, img):
        # start_sobel = time.time()
        Kx = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], cp.float32)
        Ky = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], cp.float32)

        Ix = ndimage.convolve(img, Kx)
        Iy = ndimage.convolve(img, Ky)

        G = cp.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = cp.arctan2(Iy, Ix)
        # end_sobel = time.time()
        # print(f"Time taken for sobel: {end_sobel - start_sobel} seconds")
        return G, theta

    # def non_max_suppression(self, img, D):
    #     start_maxsup = time.time()
    #     M, N = img.shape
    #     Z = cp.zeros((M, N), dtype=cp.int32)
    #     angle = D * 180. / cp.pi
    #     angle[angle < 0] += 180

    #     for i in range(1, M-1):
    #         for j in range(1, N-1):
    #             q, r = 255, 255
    #             if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
    #                 q, r = img[i, j+1], img[i, j-1]
    #             elif (22.5 <= angle[i, j] < 67.5):
    #                 q, r = img[i+1, j-1], img[i-1, j+1]
    #             elif (67.5 <= angle[i, j] < 112.5):
    #                 q, r = img[i+1, j], img[i-1, j]
    #             elif (112.5 <= angle[i, j] < 157.5):
    #                 q, r = img[i-1, j-1], img[i+1, j+1]

    #             Z[i, j] = img[i, j] if (img[i, j] >= q) and (img[i, j] >= r) else 0

    #     end_maxsup = time.time()
    #     print(f"Time taken for non max suppression: {end_maxsup - start_maxsup} seconds")
    #     return Z

    def non_max_suppression(self, img, D):
        # start_maxsup = time.time()

        M, N = img.shape
        Z = cp.zeros((M, N), dtype=cp.int32)
        angle = D * 180. / cp.pi
        angle[angle < 0] += 180

        # Define directional masks for each angle range
        angle_0 = (0 <= angle) & (angle < 22.5) | (157.5 <= angle) & (angle <= 180)
        angle_45 = (22.5 <= angle) & (angle < 67.5)
        angle_90 = (67.5 <= angle) & (angle < 112.5)
        angle_135 = (112.5 <= angle) & (angle < 157.5)

        # Perform non-maximum suppression for each angle range
        Z[1:M-1, 1:N-1] = cp.where(
            (angle_0[1:M-1, 1:N-1]) & (img[1:M-1, 1:N-1] >= img[1:M-1, 2:N]) & (img[1:M-1, 1:N-1] >= img[1:M-1, 0:N-2]),
            img[1:M-1, 1:N-1], Z[1:M-1, 1:N-1]
        )
        Z[1:M-1, 1:N-1] = cp.where(
            (angle_45[1:M-1, 1:N-1]) & (img[1:M-1, 1:N-1] >= img[2:M, 0:N-2]) & (img[1:M-1, 1:N-1] >= img[0:M-2, 2:N]),
            img[1:M-1, 1:N-1], Z[1:M-1, 1:N-1]
        )
        Z[1:M-1, 1:N-1] = cp.where(
            (angle_90[1:M-1, 1:N-1]) & (img[1:M-1, 1:N-1] >= img[2:M, 1:N-1]) & (img[1:M-1, 1:N-1] >= img[0:M-2, 1:N-1]),
            img[1:M-1, 1:N-1], Z[1:M-1, 1:N-1]
        )
        Z[1:M-1, 1:N-1] = cp.where(
            (angle_135[1:M-1, 1:N-1]) & (img[1:M-1, 1:N-1] >= img[0:M-2, 0:N-2]) & (img[1:M-1, 1:N-1] >= img[2:M, 2:N]),
            img[1:M-1, 1:N-1], Z[1:M-1, 1:N-1]
        )

        # end_maxsup = time.time()
        # print(f"Time taken for non max suppression: {end_maxsup - start_maxsup} seconds")
        return Z

    def threshold(self, img):
        # start_thresh = time.time()
        highThreshold = img.max() * self.highThreshold
        lowThreshold = highThreshold * self.lowThreshold

        M, N = img.shape
        res = cp.zeros((M, N), dtype=cp.int32)

        strong, weak = cp.int32(self.strong_pixel), cp.int32(self.weak_pixel)
        strong_i, strong_j = cp.where(img >= highThreshold)
        zeros_i, zeros_j = cp.where(img < lowThreshold)
        weak_i, weak_j = cp.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        # end_thresh = time.time()
        # print(f"Time taken for Thresholding: {end_thresh - start_thresh} seconds")
        return res

    # def hysteresis(self, img):
    #     start_hy = time.time()
    #     M, N = img.shape
    #     weak, strong = self.weak_pixel, self.strong_pixel

    #     for i in range(1, M-1):
    #         for j in range(1, N-1):
    #             if img[i, j] == weak:
    #                 if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
    #                     or (img[i, j-1] == strong) or (img[i, j+1] == strong)
    #                     or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
    #                     img[i, j] = strong
    #                 else:
    #                     img[i, j] = 0
    #     end_hy = time.time()
    #     print(f"Time taken for hysteresis: {end_hy - start_hy} seconds")
    #     return img

    def hysteresis(self, img):
        # start_hy = time.time()
        weak, strong = self.weak_pixel, self.strong_pixel
        
        # Pad the image to handle border pixels without explicit loops
        padded_img = cp.pad(img, pad_width=1, mode='constant', constant_values=0)
        
        # Initial mask of strong pixels
        strong_pixels = (padded_img == strong)
        
        for _ in range(3):  # Iteratively propagate strong edges; 3 iterations should suffice
            # Mask of weak pixels in the inner area of padded_img
            weak_pixels = (padded_img[1:-1, 1:-1] == weak)
            
            # Check surrounding 8 pixels in the padded image for any strong pixels
            connected_to_strong = (
                strong_pixels[1:-1, :-2] | strong_pixels[1:-1, 2:] |      # Left and right
                strong_pixels[:-2, 1:-1] | strong_pixels[2:, 1:-1] |      # Top and bottom
                strong_pixels[:-2, :-2] | strong_pixels[:-2, 2:] |        # Top-left and top-right
                strong_pixels[2:, :-2] | strong_pixels[2:, 2:]            # Bottom-left and bottom-right
            )
            
            # Update weak pixels that are connected to strong pixels
            padded_img[1:-1, 1:-1] = cp.where(weak_pixels & connected_to_strong, strong, padded_img[1:-1, 1:-1])
            
            # Update strong pixels for the next iteration
            strong_pixels = (padded_img == strong)
        
        # Remove padding and finalize the image
        result = cp.where(padded_img[1:-1, 1:-1] != weak, padded_img[1:-1, 1:-1], 0)

        # end_hy = time.time()
        # print(f"Time taken for hysteresis: {end_hy - start_hy} seconds")
        return result


    def detect(self):
        for img in self.imgs:
            img_smoothed = ndimage.convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            gradientMat, thetaMat = self.sobel_filters(img_smoothed)
            nonMaxImg = self.non_max_suppression(gradientMat, thetaMat)
            thresholdImg = self.threshold(nonMaxImg)
            img_final = self.hysteresis(thresholdImg)
            # start = time.time()
            self.imgs_final.append(cp.asnumpy(img_final))  # Convert back to numpy for output
            # end = time.time()

            # print(f"Time taken to send to cpu: {end-start} seconds")

        return self.imgs_final
