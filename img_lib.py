# img_lib.py
# Lau Yan Han (2022)
# 
# Library of functions to operate on images across multiple cameras
# Use this library for Re-ID algorithms that operate on instantaneous
# snapshots of detected targets.

import csv
import cv2
import numpy as np
import img_parameters as parm

# extract data from csv file
def extract_data(src):
    extracted = []
    with open (src, 'r') as csvfile:
        csvreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in csvreader:
            extracted.append(row)
    return extracted

# get angle (radian) between two vectors a and b (represented as arrays)
def get_angle(a, b):
    dot_product = np.dot(a, b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(dot_product / norm_product)

# return intersection of two 2d lines in x-y coordinates (ndarray format)
# input: endpoints of both lines (x-y ndarray format)
# https://www.geeksforgeeks.org/program-for-point-of-intersection-of-two-lines/
def two_lines_intersection(line1_ptA, line1_ptB, line2_ptA, line2_ptB):
    x_1A = line1_ptA[0]
    y_1A = line1_ptA[1]
    x_1B = line1_ptB[0]
    y_1B = line1_ptB[1]
    x_2A = line2_ptA[0]
    y_2A = line2_ptA[1]
    x_2B = line2_ptB[0]
    y_2B = line2_ptB[1]

    # Line 1 represented as a1x + b1y = c1
    a1 = y_1B - y_1A
    b1 = x_1A - x_1B
    c1 = a1*x_1A + b1*y_1A

    # Line 2 represented as a2x + b2y = c2
    a2 = y_2B - y_2A
    b2 = x_2A - x_2B
    c2 = a2*x_2A + b2*y_2A

    det = a1*b2 - a2*b1
    if det:
        x = (b2*c1 - b1*c2)/det
        y = (a1*c2 - a2*c1)/det
    else:
        # lines are parallel if determinant = 0
        x = 1e10
        y = 1e10
    return np.array([x, y])

# return cross ratio of 4 collinear 2D points (represented as arrays)
def cross_ratio(a, b, c, d):
    ac = np.linalg.norm(np.subtract(a,c))
    bd = np.linalg.norm(np.subtract(b,d))
    bc = np.linalg.norm(np.subtract(b,c))
    ad = np.linalg.norm(np.subtract(a,d))
    return (ac*bd)/(bc*ad)

# place original images side by side, annotate them with re-id results and save combined image
# inputs:
# 1. Images (CV Mat) and target centroids (ndarray) of Cameras 0 and 1
# 2. Target IDs of Camera 0 (ndarray)
# 3. Matrix of similarity scores for all pairs of targets across camers (ndarray)
# 4. Whether to use max or min similarity score to Re-ID targets (boolean)
def annotate_and_save_reid(img0, img1, centroids0, centroids1, IDs_0, similarity_mat, use_max):
    
    # "matched" - for each target ID in Camera 0 (represented by the row of "matched"), 
    # get the ID of the matched target in Camera 1 (represented by the value in "matched")
    if use_max:
        matched = np.argmax(similarity_mat, axis=1)
    else:
        matched = np.argmin(similarity_mat, axis=1)

    # annotate re-id images
    combined_img = cv2.hconcat([img0, img1])

    for id0 in IDs_0:
        # Camera 0 "id0" is matched with Camera 1 "id1"
        id1 = matched[id0]

        # generate target centroid coordinates to be annotated on the frame
        id0_coords = centroids0[id0].copy()
        # transform from standard coords (origin at bottom left) to img coords (origin at top left)
        id0_coords[1] = parm.IMG_HEIGHT - id0_coords[1]
        # convert centroid floats to int
        id0_coords = [int(x) for x in id0_coords]
        # convert to tuple to be compatible with OpenCV line
        id0_coords = tuple(id0_coords)

        id1_coords = centroids1[id1].copy()
        id1_coords[1] = parm.IMG_HEIGHT - id1_coords[1]
        # transpose to the right for Camera 1 IDs
        id1_coords[0] += parm.IMG_WIDTH
        id1_coords = [int(x) for x in id1_coords]
        id1_coords = tuple(id1_coords)

        combined_img = cv2.line(combined_img, id0_coords, id1_coords, parm.colours[id0%10], 3)

    cv2.imwrite('data/pics/saved.jpg', combined_img)

# base class containing csv, img and target data for each camera
class camera_base():
    
    def __init__(self, cam_ID, csv_file, img) -> None:

        self.cam_ID = cam_ID
        self.csv_data = extract_data(csv_file)
        self.img = cv2.imread(img)

        # basic target data. IDs are zero indexed
        self.num_of_targets = len(self.csv_data)
        self.IDs = list(range(0,self.num_of_targets))
        self.areas = np.ones(self.num_of_targets)
        self.centroids = np.ones([self.num_of_targets,2])
        self.get_target_areas_centroids()

    # assign target IDs and extract areas + XY centroids from CSV data
    # convert from image convention (origin at top left)
    # to standard XY convention (origin at bottom left)
    def get_target_areas_centroids(self):
        id = 0
        for row in self.csv_data:
            width = abs(row[parm.XMAX] - row[parm.XMIN])
            height = abs(row[parm.YMAX] - row[parm.YMIN])
            self.areas[id] = width*height
            cen_x = 0.5*(row[parm.XMAX] + row[parm.XMIN])
            cen_y = parm.IMG_HEIGHT - 0.5*(row[parm.YMAX] + row[parm.YMIN])
            self.centroids[id][0] = cen_x
            self.centroids[id][1] = cen_y
            id += 1