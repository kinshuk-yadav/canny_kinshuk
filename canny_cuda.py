# import cupy as cp
# from cupyx.scipy import ndimage

# class CannyEdgeDetectorCUDA:
#     def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
#         self.imgs = [cp.array(img) for img in imgs]
#         self.imgs_final = []
#         self.weak_pixel = weak_pixel
#         self.strong_pixel = strong_pixel
#         self.sigma = sigma
#         self.kernel_size = kernel_size
#         self.lowThreshold = lowthreshold
#         self.highThreshold = highthreshold

#     def gaussian_kernel(self, size, sigma=1):
#         size = int(size) // 2
#         x, y = cp.mgrid[-size:size+1, -size:size+1]
#         normal = 1 / (2.0 * cp.pi * sigma**2)
#         g = cp.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
#         return g

#     def sobel_filters(self, img):
#         Kx = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], cp.float32)
#         Ky = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], cp.float32)

#         Ix = ndimage.convolve(img, Kx)
#         Iy = ndimage.convolve(img, Ky)

#         G = cp.hypot(Ix, Iy)
#         G = G / G.max() * 255
#         theta = cp.arctan2(Iy, Ix)
#         return G, theta

#     def non_max_suppression(self, img, D):
#         M, N = img.shape
#         Z = cp.zeros((M, N), dtype=cp.int32)
#         angle = D * 180. / cp.pi
#         angle[angle < 0] += 180

#         for i in range(1, M-1):
#             for j in range(1, N-1):
#                 q, r = 255, 255
#                 if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
#                     q, r = img[i, j+1], img[i, j-1]
#                 elif (22.5 <= angle[i, j] < 67.5):
#                     q, r = img[i+1, j-1], img[i-1, j+1]
#                 elif (67.5 <= angle[i, j] < 112.5):
#                     q, r = img[i+1, j], img[i-1, j]
#                 elif (112.5 <= angle[i, j] < 157.5):
#                     q, r = img[i-1, j-1], img[i+1, j+1]

#                 Z[i, j] = img[i, j] if (img[i, j] >= q) and (img[i, j] >= r) else 0

#         return Z

#     def threshold(self, img):
#         highThreshold = img.max() * self.highThreshold
#         lowThreshold = highThreshold * self.lowThreshold

#         M, N = img.shape
#         res = cp.zeros((M, N), dtype=cp.int32)

#         strong, weak = cp.int32(self.strong_pixel), cp.int32(self.weak_pixel)
#         strong_i, strong_j = cp.where(img >= highThreshold)
#         zeros_i, zeros_j = cp.where(img < lowThreshold)
#         weak_i, weak_j = cp.where((img <= highThreshold) & (img >= lowThreshold))

#         res[strong_i, strong_j] = strong
#         res[weak_i, weak_j] = weak

#         return res

#     def hysteresis(self, img):
#         M, N = img.shape
#         weak, strong = self.weak_pixel, self.strong_pixel

#         for i in range(1, M-1):
#             for j in range(1, N-1):
#                 if img[i, j] == weak:
#                     if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
#                         or (img[i, j-1] == strong) or (img[i, j+1] == strong)
#                         or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
#                         img[i, j] = strong
#                     else:
#                         img[i, j] = 0

#         return img

#     def detect(self):
#         for img in self.imgs:
#             img_smoothed = ndimage.convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
#             gradientMat, thetaMat = self.sobel_filters(img_smoothed)
#             nonMaxImg = self.non_max_suppression(gradientMat, thetaMat)
#             thresholdImg = self.threshold(nonMaxImg)
#             img_final = self.hysteresis(thresholdImg)
#             self.imgs_final.append(cp.asnumpy(img_final))  # Convert back to numpy for output

#         return self.imgs_final




import cupy as cp
from cupyx.scipy import ndimage
import time

class CannyEdgeDetectorCUDA:
    # def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
    #     # Load images as GPU arrays
    #     self.imgs = [cp.array(img) for img in imgs]
    #     self.imgs_final = []
    #     self.weak_pixel = weak_pixel
    #     self.strong_pixel = strong_pixel
    #     self.sigma = sigma
    #     self.kernel_size = kernel_size
    #     self.lowThreshold = lowthreshold
    #     self.highThreshold = highthreshold

    # def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
    #     self.imgs = [cp.array(img) for img in imgs]
    #     self.imgs_final = []
    #     self.weak_pixel = weak_pixel
    #     self.strong_pixel = strong_pixel
    #     self.sigma = sigma
    #     self.kernel_size = kernel_size
    #     self.lowThreshold = lowthreshold
    #     self.highThreshold = highthreshold

    # def gaussian_kernel(self, size, sigma=1):
    #     size = int(size) // 2
    #     x, y = cp.mgrid[-size:size+1, -size:size+1]
    #     normal = 1 / (2.0 * cp.pi * sigma**2)
    #     g = cp.exp(-((x*2 + y2) / (2.0 * sigma*2))) * normal
    #     return g

    # def sobel_filters(self, img):
    #     Kx = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], cp.float32)
    #     Ky = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], cp.float32)

    #     Ix = ndimage.convolve(img, Kx)
    #     Iy = ndimage.convolve(img, Ky)

    #     G = cp.hypot(Ix, Iy)
    #     G = G / G.max() * 255
    #     theta = cp.arctan2(Iy, Ix)
    #     return G, theta

    # def non_max_suppression(self, img, D):
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

    #     return Z

    # def threshold(self, img):
    #     highThreshold = img.max() * self.highThreshold
    #     lowThreshold = highThreshold * self.lowThreshold

    #     M, N = img.shape
    #     res = cp.zeros((M, N), dtype=cp.int32)

    #     strong, weak = cp.int32(self.strong_pixel), cp.int32(self.weak_pixel)
    #     strong_i, strong_j = cp.where(img >= highThreshold)
    #     zeros_i, zeros_j = cp.where(img < lowThreshold)
    #     weak_i, weak_j = cp.where((img <= highThreshold) & (img >= lowThreshold))

    #     res[strong_i, strong_j] = strong
    #     res[weak_i, weak_j] = weak

    #     return res

    # def hysteresis(self, img):
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

    #     return img

    # def detect(self):
    #     # Initialize GPU timing events
    #     start_event = cp.cuda.Event()
    #     end_event = cp.cuda.Event()

    #     # Start GPU timing
    #     start_event.record()

    #     # Process each image on the GPU
    #     for img in self.imgs:
    #         img_smoothed = ndimage.convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
    #         gradientMat, thetaMat = self.sobel_filters(img_smoothed)
    #         nonMaxImg = self.non_max_suppression(gradientMat, thetaMat)
    #         thresholdImg = self.threshold(nonMaxImg)
    #         img_final = self.hysteresis(thresholdImg)
    #         self.imgs_final.append(img_final)  # Remain on GPU

    #     # End GPU timing
    #     end_event.record()
    #     end_event.synchronize()  # Wait for GPU to finish

    #     # Calculate and print GPU processing time (excluding data transfer)
    #     gpu_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    #     print(f"GPU Processing Time (excluding data transfer): {gpu_time_ms / 1000} seconds")

    #     # Convert images back to CPU (only if necessary)
    #     self.imgs_final = [cp.asnumpy(img) for img in self.imgs_final]
    #     printf("No errors here")
    #     return self.imgs_final

    def __init__(self, imgs, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
        # Load images as GPU arrays
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
        Kx = cp.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], cp.float32)
        Ky = cp.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], cp.float32)

        Ix = ndimage.convolve(img, Kx)
        Iy = ndimage.convolve(img, Ky)

        G = cp.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = cp.arctan2(Iy, Ix)
        return G, theta

    def non_max_suppression(self, img, D):
        M, N = img.shape
        Z = cp.zeros((M, N), dtype=cp.int32)
        angle = D * 180. / cp.pi
        angle[angle < 0] += 180

        for i in range(1, M-1):
            for j in range(1, N-1):
                q, r = 255, 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q, r = img[i, j+1], img[i, j-1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q, r = img[i+1, j-1], img[i-1, j+1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q, r = img[i+1, j], img[i-1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q, r = img[i-1, j-1], img[i+1, j+1]

                Z[i, j] = img[i, j] if (img[i, j] >= q) and (img[i, j] >= r) else 0

        return Z

    def threshold(self, img):
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

        return res

    def hysteresis(self, img):
        M, N = img.shape
        weak, strong = self.weak_pixel, self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if img[i, j] == weak:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0

        return img

    def detect(self):
        # Initialize GPU timing events
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        # Start GPU timing
        start_event.record()

        # Process each image on the GPU
        for img in self.imgs:
            img_smoothed = ndimage.convolve(img, self.gaussian_kernel(self.kernel_size, self.sigma))
            gradientMat, thetaMat = self.sobel_filters(img_smoothed)
            nonMaxImg = self.non_max_suppression(gradientMat, thetaMat)
            thresholdImg = self.threshold(nonMaxImg)
            img_final = self.hysteresis(thresholdImg)
            self.imgs_final.append(img_final)  # Remain on GPU

        # End GPU timing
        end_event.record()
        end_event.synchronize()  # Wait for GPU to finish

        # Calculate and print GPU processing time (excluding data transfer)
        gpu_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        print(f"GPU Processing Time (excluding data transfer): {gpu_time_ms / 1000} seconds")

        # Convert images back to CPU (only if necessary)
        self.imgs_final = [cp.asnumpy(img) for img in self.imgs_final]
        
        return self.imgs_final
