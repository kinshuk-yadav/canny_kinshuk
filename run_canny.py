# run_canny.py

from utils import visualize, load_data, load_with_noise

import cv2
import numpy as np
from canny_try import cannyEdgeDetector
from improved_canny_try import ImprovedCannyEdgeDetector
from cuda_cpu import CannyEdgeDetectorCUDA
from improved_canny_cuda import ImprovedCannyEdgeDetectorCUDA
import time

def main():
    # Load the image in grayscale
    print("hi")
    # image_path = './lenna.jpeg'
    # # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    imgs = load_with_noise(percentage_of_noise=0.01)
    # imgs = load_data()

    # if img is None:
    #     print(f"Could not load image from path: {image_path}")
    #     return

    # visualize(img, 'gray')

    # Initialize the Canny edge detector with the image
    edge_detector = cannyEdgeDetector(imgs)
    imp = ImprovedCannyEdgeDetector(imgs)
    edge_detector_cuda = CannyEdgeDetectorCUDA(imgs)
    imp_cuda = ImprovedCannyEdgeDetectorCUDA(imgs)
    
    # Run the edge detection
    start_time_edges = time.time()
    edges = edge_detector.detect()
    end_time_edges = time.time()

    start_time_edges2 = time.time()
    edges2 = imp.detect()
    end_time_edges2 = time.time()

    start_time_edges_cuda = time.time()
    edges_cuda = edge_detector_cuda.detect()
    end_time_edges_cuda = time.time()

    start_time_edges4 = time.time()
    edges4 = imp_cuda.detect()
    end_time_edges4 = time.time()

    print(f"Time taken by CPU for Canny Edge Detector: {end_time_edges - start_time_edges} seconds")
    print(f"Time taken by CPU for Improved Canny Edge Detector: {end_time_edges2 - start_time_edges2} seconds")
    print(f"Time taken by GPU for Canny Edge Detector: {end_time_edges_cuda - start_time_edges_cuda} seconds")
    print(f"Time taken by GPU for Improved Canny Edge Detector: {end_time_edges4 - start_time_edges4} seconds")
    
    # img.append(edges)
    # imgs.append(edges)
    # imgs = imgs + edges + edges2 + edges_cuda 
    imgs = imgs + edges + edges2 + edges_cuda + edges4
    visualize(imgs, 'gray')
    
    # Convert the result to uint8 for display compatibility
    # edges_uint8 = np.uint8(edges[0])

    # # Display the output image (edges detected)
    # cv2.imshow("Canny Edge Detection", edges_uint8)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
