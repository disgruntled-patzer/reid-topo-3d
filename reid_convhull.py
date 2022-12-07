# reid_convhull.py
# Lau Yan Han and Lua Chong Ghee (2022)
#
# Overview:
# Perform re-identification of targets across two images using the method 
# described in Lua et al (2022): https://ieeexplore.ieee.org/abstract/document/9931699
# 
# Inputs: 
# 1. csv files with detected targets (one for each target). Stored as
# a size 2 list in variable csv_files, one cell for each camera
# 2. original image files. Stored as a size 2 list in variable img_files,
# one cell for each camera. Use for visualisation purposes later

import img_lib
import matplotlib.pyplot as plt
import numpy as np
import img_parameters as parm
from scipy.spatial import ConvexHull

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

class camera (img_lib.camera_base):

    # initialise parameters from parent class
    def __init__(self, cam_ID, csv_file, img) -> None:
        super().__init__(cam_ID, csv_file, img)
    
    # generate convex hull of detected targets and initialise list of cross ratios
    def generate_convexhull(self):
        self.hull = ConvexHull(self.centroids)
        self.hull_nvertices = len(self.hull.vertices)
        if self.hull_nvertices < 5:
            raise Exception("Algorithm needs 5 or more convex hull vertices")
        self.cross_ratio_seq = np.zeros(self.hull_nvertices)
    
    # Given a convex hull vertex's position, get the target ID and centroids of the vertex and its 4 neighbours
    def get_vertex_and_neighbour_data(self, hull_pos):
        if hull_pos >= self.hull_nvertices:
            raise Exception("Position exceeds number of convex hull vertices")
        # neighbours: Previous two vertices, and next two vertices
        positions = np.array([hull_pos - 2, hull_pos - 1, hull_pos, hull_pos + 1, hull_pos + 2])
        # wraparound the number of convex hull vertices
        positions = np.mod(positions, self.hull_nvertices)
        # given the positions, recover the original target IDs and x-y centroids of these vertices
        target_IDs = np.zeros(5, dtype=int)
        target_centroids = np.zeros([5,2])
        for i in range(5):
            target_IDs[i] = self.hull.vertices[positions[i]]
            target_centroids[i] = self.centroids[target_IDs[i]]  
        return target_IDs, target_centroids
    
    # calculate the cross ratio for a given vertex's position on the convex hull
    def cross_ratio_single_vertex(self, hull_pos):
        _, target_centroids = self.get_vertex_and_neighbour_data(hull_pos)
        # get key points
        pt1 = target_centroids[1]
        pt2 = two_lines_intersection(target_centroids[0], target_centroids[2],
                                    target_centroids[1], target_centroids[3])
        pt3 = two_lines_intersection(target_centroids[1], target_centroids[3],
                                    target_centroids[2], target_centroids[4])
        pt4 = target_centroids[3]
        return cross_ratio(pt1, pt2, pt3, pt4)
    
    # calculate cross ratio for all vertices on the convex hull
    def cal_cross_ratio_seq(self):
        for hull_pos in range(self.hull_nvertices):
            self.cross_ratio_seq[hull_pos] = self.cross_ratio_single_vertex(hull_pos)
    
    # takes a convex vertex position (pos) and returns cross ratios of the remaining sequence
    # in the following order: pos+1, pos+2 ... and wraparound to 0, 1, ... pos-1
    def get_remaining_seq_cross_ratios(self, hull_pos):
        return np.concatenate((self.cross_ratio_seq[hull_pos+1:], self.cross_ratio_seq[:hull_pos]))

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('X')
ax.set_ylabel('Y')

cameras = (camera(0, parm.csv_files[0], parm.img_files[0]), 
            camera(1, parm.csv_files[1], parm.img_files[1]))
for cam in cameras:
    # generate initial data
    cam.get_target_areas_centroids()
    cam.generate_convexhull()
    cam.cal_cross_ratio_seq()
    # visualise convex hulls
    if cam.cam_ID:
        m = 'o'
    else:
        m = '^'
    plt.scatter(cam.centroids[:,0], cam.centroids[:,1], marker=m)
    for simplex in cam.hull.simplices:
        plt.plot(cam.centroids[simplex,0], cam.centroids[simplex,1], 'k-')

# associate across cameras
assoc_err = 1e5*np.ones([cameras[0].num_of_targets, cameras[1].num_of_targets])
for pos_0, ID_0 in enumerate(cameras[0].hull.vertices):
    cross_ratio_seq_0 = cameras[0].get_remaining_seq_cross_ratios(pos_0)
    for pos_1, ID_1 in enumerate(cameras[1].hull.vertices):
        cross_ratio_seq_1 = cameras[1].get_remaining_seq_cross_ratios(pos_1)
        cross_ratio_err = np.abs(np.subtract(cross_ratio_seq_1, cross_ratio_seq_0))
        assoc_err[ID_0][ID_1] = np.mean(cross_ratio_err)

# Re-ID visualisation
img_lib.annotate_and_save_reid(cameras[0].img, cameras[1].img,
                                cameras[0].centroids, cameras[1].centroids,
                                cameras[0].IDs, assoc_err, use_max=False)

plt.show()