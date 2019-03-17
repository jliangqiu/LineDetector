import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt


files = ["1.jpeg", "2.jpeg", "3.jpeg", "4.jpeg", "5.jpeg"]
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 30  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 100  # minimum number of pixels making up a line
max_line_gap = 15  # maximum gap in pixels between connectable line segments

# canny params
low_threshold = 50
high_threshold = 150

if __name__ == "__main__":
    for filein in files:
        # white color mask
        img = cv2.imread(filein, 0)
        result = img.copy()
        # gaussian
        kernel_size = 5
        img = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)
        #converted = convert_hls(img)

        edges = cv2.Canny(img, low_threshold, high_threshold)
        cv2.imshow("edges_" + filein,edges)
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(result,(x1,y1),(x2,y2),(255,0,0), 1)
        cv2.imshow("mask_" + str(filein), result)
    cv2.waitKey(0)