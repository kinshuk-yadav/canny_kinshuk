# run_canny.py

from utils import visualize, load_data, load_with_noise, temp_visualize

import cv2
import numpy as np
from canny_try import cannyEdgeDetector
from improved_canny_try import ImprovedCannyEdgeDetector

import time

def main():
    # Load the image in grayscale
    print("Execution times of Canny Edge Detector:\n")
    # image_path = './lenna.jpeg'
    # # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    imgs = load_with_noise(percentage_of_noise=0.01)
    # imgs = load_data()

    # Initialize the Canny edge detector with the image
    edge_detector = cannyEdgeDetector(imgs)
    imp = ImprovedCannyEdgeDetector(imgs)

    
    # Run the edge detection
    start_time_edges = time.time()
    edges = edge_detector.detect()
    end_time_edges = time.time()

    start_time_edges2 = time.time()
    with np.errstate(divide='ignore', invalid='ignore'):
        edges2 = imp.detect()
    end_time_edges2 = time.time()


    print(f"Time taken by CPU for Canny Edge Detector: {end_time_edges - start_time_edges} seconds")
    print(f"Time taken by CPU for Improved Canny Edge Detector: {end_time_edges2 - start_time_edges2} seconds")


    imgs = imgs + edges + edges2
    temp_visualize(imgs, 'gray')

if __name__ == "__main__":
    main()
