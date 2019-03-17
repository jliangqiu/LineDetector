import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import random
import sys
import math

files = ["in/1.jpeg", "in/2.jpeg", "in/3.jpeg", "in/4.jpeg", "in/5.jpeg"]

# canny params
low_threshold = 100
high_threshold = 150
 
# ransac params

threshold_points = 25 # how many points in neighbouring distance in order to be considered a valid line
threshold_distance = 2.2 # distance form the line to be taken into account
diff_margin_angle = 0.2 # difference between angles to be considered a different line from those already taken
diff_margin_y_intercept = 20 # different for y intercept to be considered a different line from those already taken
nr_samples = 2000
max_diruption = 2.2

def find_line_model(points):
    """ find a line model for the given points
    :param points selected points for model fitting
    :return line model
    """
 
    # [WARNING] vertical and horizontal lines should be treated differently
    #           here we just add some noise to avoid division by zero
 
    # find a line model for these points
    m = (points[1,1] - points[0,1]) / (points[1,0] - points[0,0] + sys.float_info.epsilon)  # slope (gradient) of the line
    c = points[1,1] - m * points[1,0]                                     # y-intercept of the line
 
    return m, c

def find_intercept_point(m, c, x0, y0):
    """ find an intercept point of the line model with
        a normal from point (x0,y0) to it
    :param m slope of the line model
    :param c y-intercept of the line model
    :param x0 point's x coordinate
    :param y0 point's y coordinate
    :return intercept point
    """
 
    # intersection point with the model
    x = (x0 + m*y0 - m*c)/(1 + m**2)
    y = (m*x0 + (m**2)*y0 - (m**2)*c)/(1 + m**2) + c
 
    return x, y

def get_core_line(inliers, m, c, max_diruption=4, threshold_points=100):

    start = 0
    count_consecutive = 0
    lines = []
    for i in range(1, len(inliers)):
        if (math.fabs(inliers[i-1][1] - inliers[i][1]) > max_diruption) or (i == len(inliers) - 1):
            if count_consecutive > threshold_points:
                lines.append( (m, c, np.arctan(m), inliers[start][0], inliers[i-1][0]) )
            start = i
            count_consecutive = 0
        else:
            count_consecutive += 1
    return lines

# decide if the 2 points give a line or not (by computing how many points are near the line)
# the current line can by splitted (max disrutpion) in many segments, (get core lines)
def fit(data, m, c, x2, y2, threshold_points, threshold_distance, max_diruption):
    inliers = []
    for ind in range(len(data)):
 
        x0 = data[ind][0]
        y0 = data[ind][1]
 
        # find an intercept point of the model with a normal from point (x0,y0)
        x1, y1 = find_intercept_point(m, c, x0, y0)
 
        # distance from point to the model
        dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
 
        # check whether it's an inlier or not
        if dist < threshold_distance:
            d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if x1 < x2:
                d = -d 
            inliers.append( ((x0, y0), d))
    # get pure lines
    if len(inliers) > threshold_points:
        sorted_inliers = sorted(inliers, key=lambda x: x[1])
        lines = get_core_line(sorted_inliers, m, c, max_diruption, threshold_points)
        if len(lines) > 0:
            print(lines)
        return lines
    return []

def taken(lines, p11, p12, diff_margin_angle, diff_margin_y_intercept):
    radians = np.arctan(m)
    for line in lines:
        p21 = line[3]
        p22 = line[4]
        d11 = math.sqrt((p11[0] - p21[0])**2 + (p11[1] - p21[1])**2)
        d12 = math.sqrt((p11[0] - p22[0])**2 + (p11[1] - p22[1])**2)
        d21 = math.sqrt((p12[0] - p21[0])**2 + (p12[1] - p21[1])**2)
        d22 = math.sqrt((p12[0] - p22[0])**2 + (p12[1] - p22[1])**2)
        distances = [d11, d12, d22, d21]
        for d1 in distances:
            for d2 in distances:
                if d1 != d2 and d1 * d2 < 30:
                    return True
    return False

def ransacc(data, threshold_points=200, threshold_distance=8, nr_samples=1000, diff_margin_angle=0.05, diff_margin_y_intercept=20, max_diruption=4):
    # perform RANSAC iterations
    lines = []
    # take 2 points
    for it in range(nr_samples):
        ind1 = random.randint(0, len(data) - 1)
        ind2 = random.randint(0, len(data) - 1)
        (x1, y1) = data[ind1]
        (x2, y2) = data[ind2]
    
        # find a line model for these points
        m, c = find_line_model(np.array([[x1, y1], [x2, y2]]))
        
        lines_new = fit(data, m , c, x1, y1, threshold_points, threshold_distance, max_diruption)
        # check if already taken
        for line in lines_new:
            if taken(lines, line[3], line[4], diff_margin_angle, diff_margin_y_intercept) == False:
                lines.append(line)
    return lines
    
if __name__ == "__main__":
    random.seed(None)

    for filein in files[4:5]:
        # white color mask
        img = cv2.imread(filein, 0)
        result = img.copy()
        # gaussian
        kernel_size = 5
        img = cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

        edges = cv2.Canny(img, low_threshold, high_threshold)
        cv2.imshow("edges_" + filein, edges)

        n = len(edges)
        m = len(edges[0])
        data = []
        cnt = 0
        for i in range(len(edges)):
            for j in range(len(edges[i])):
                if edges[i][j] == 255:
                    data.append((j, len(edges) - i - 1))
        lines = ransacc(data, threshold_points, threshold_distance, nr_samples, diff_margin_angle, diff_margin_y_intercept, max_diruption)
        print('number of lines found {} form {}'.format(len(lines), nr_samples))
        for line in lines:
            (x1, y1) = line[3]
            (x2, y2) = line[4]
            y1 = len(edges) - y1
            y2 = len(edges) - y2

            cv2.line(result,(x1,y1),(x2,y2),(255,0,0), 1)
        cv2.imshow("lines_" + filein, result)
    cv2.waitKey(0)