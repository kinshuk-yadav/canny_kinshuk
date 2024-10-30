# run_canny.py

from utils import visualize, load_data

import cv2
import numpy as np
from canny_try import cannyEdgeDetector

def main():
    # Load the image in grayscale
    print("hi")
    image_path = './lenna.jpeg'
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    imgs = load_data()

    # if img is None:
    #     print(f"Could not load image from path: {image_path}")
    #     return

    # visualize(img, 'gray')

    # Initialize the Canny edge detector with the image
    edge_detector = cannyEdgeDetector(imgs)
    
    # Run the edge detection
    edges = edge_detector.detect()
    # img.append(edges)
    # imgs.append(edges)
    imgs = imgs + edges
    visualize(imgs, 'gray')
    
    # Convert the result to uint8 for display compatibility
    # edges_uint8 = np.uint8(edges[0])

    # # Display the output image (edges detected)
    # cv2.imshow("Canny Edge Detection", edges_uint8)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
