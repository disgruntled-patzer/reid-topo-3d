# reid_delaunay_3d.py
# Lau Yan Han (2022)
#
# Overview:
# Similar to reid_delaunay.py. This time, relative 3D coordinates
# are estimated for each target (relative to the first detected target
# for each camera) before 3D Delaunay Triangles are generated. Then the
# re-identification process and output is similar to reid_delaunay.py

import csv
# import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Delaunay

##############################
# data format saved by Yolo V5
##############################

# id    xmin    ymin    xmax   ymax  confidence  class    name
#  0  749.50   43.50  1148.0  704.5    0.874023      0  person
#  1  433.50  433.50   517.5  714.5    0.687988     27     tie
#  2  114.75  195.75  1095.0  708.0    0.624512      0  person
#  3  986.00  304.00  1028.0  420.0    0.286865     27     tie
# (xmin,ymin) is top-left, (xmax,ymax) is lower right. (0,0) is top-left of img

##########################
# functions and parameters
##########################

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
XMIN = 1
YMIN = 2
XMAX = 3
YMAX = 4
csv_files = [
    'data/csv/bsp_0.csv',
    'data/csv/bsp_1.csv'
]

# get angle (radian) between two vectors a and b (represented as arrays)
def get_angle(a, b):
    dot_product = np.dot(a, b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(dot_product / norm_product)

# solid angle about Point O in a tetrahedral OABC using L'Huilier's Thm
def solid_angle(theta_a, theta_b, theta_c):
    theta_s = 0.5*(theta_a + theta_b + theta_c)
    temp = math.sqrt(math.tan(0.5*theta_s) * math.tan(0.5*(theta_s - theta_a)) * \
        math.tan(0.5*(theta_s - theta_b)) * math.tan(0.5*(theta_s - theta_c)))
    return 4*math.atan(temp)

# extract data from csv file
def extract_data(src):
    extracted = []
    with open (src, 'r') as csvfile:
        csvreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in csvreader:
            extracted.append(row)
    return extracted

#############################
# camera and topology classes
#############################

class camera:

    # extract data from csv files and generate other parameters
    def __init__(self, cam_ID, csv_file) -> None:

        # csv data
        self.cam_ID = cam_ID
        self.csv_data = extract_data(csv_file)

        # basic target data. IDs are zero indexed
        self.num_of_targets = len(self.csv_data)
        self.IDs = list(range(0,self.num_of_targets))
        self.areas = np.ones(self.num_of_targets)
        self.centroids = np.ones([self.num_of_targets,2])
        self.coords3d = np.zeros([self.num_of_targets,3])

        # topological data
        self.delaunay_map = []
    
    # assign target IDs and extract areas + XY centroids from CSV data
    # convert from image convention (origin at top left)
    # to standard XY convention (origin at bottom left)
    def get_target_areas_centroids(self):
        id = 0
        for row in self.csv_data:
            width = abs(row[XMAX] - row[XMIN])
            height = abs(row[YMAX] - row[YMIN])
            self.areas[id] = width*height
            cen_x = 0.5*(row[XMAX] + row[XMIN])
            cen_y = IMG_HEIGHT - 0.5*(row[YMAX] + row[YMIN])
            self.centroids[id][0] = cen_x
            self.centroids[id][1] = cen_y
            id += 1
    
    # generate 3d coords of all centroids relative to 1st target
    def generate_3d_coords(self):
        
        base_x = self.centroids[0][0]
        base_y = self.centroids[0][1]
        base_area = self.areas[0]
        for id in range(len(self.centroids)):
            if id == 0:
                continue
            self.coords3d[id][0] = self.centroids[id][0] - base_x
            self.coords3d[id][1] = self.centroids[id][1] - base_y
            area_ratio = self.areas[id]/base_area
            self.coords3d[id][2] = (1/area_ratio) - 1 # 3d path re-id, depth est formula

    # generate map of delaunay triangles for all targets in camera
    def generate_delaunay(self):
        self.delaunay_map = Delaunay(self.coords3d)
        # print(self.delaunay_map.simplices)

###############
# main pipeline
###############

fig = plt.figure(figsize=plt.figaspect(0.5))

# extract info on detected objects
cameras = (camera(0, csv_files[0]), camera(1, csv_files[1]))
for cam in cameras:
    cam.get_target_areas_centroids()
    cam.generate_3d_coords()
    cam.generate_delaunay()

    ax = fig.add_subplot(1, 2, cam.cam_ID+1, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot_trisurf(cam.coords3d[:,0], cam.coords3d[:,1], \
        cam.coords3d[:,2], triangles=cam.delaunay_map.simplices)

plt.show()