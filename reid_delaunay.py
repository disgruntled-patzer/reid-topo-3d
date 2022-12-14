# reid_delaunay.py
# Lau Yan Han (2022)
#
# Overview:
# Generate 2D topological map of targets in two camera images
# using the triangular topological sequence in Xudong et al (2022).
#
# Then, perform re-identification and generate the matrix of similarity
# scores across targets.
#
# Input: csv files with detected targets (one for each target). Stored as
# a size 2 list in variable csv_files, one cell for each camera
#
# Output: visualisation of re-id results across both images - saved in
# 'data/pics' folder and displayed in a new window
#
# Key parameter: VISUALISE_3D - Set to true to estimate 3D coordinates and use
# then for re-identification. Otherwise, the 2D coordinates will be used.

import img_lib
import img_parameters as parm
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Delaunay

#############################
# camera and topology classes
#############################

# a single delaunay triangle with the base ID (0,0,0)
# and relative coordinates of the two neighbouring vertexes
class delaunay_triangle:
    def __init__(self, ID, neighbours):
        self.base_ID_ = ID
        self.neighbours_ = neighbours
    
    def get_base_ID(self):
        return self.base_ID_
    
    def get_base_angle(self):
        return img_lib.get_angle(self.neighbours_[0], self.neighbours_[1])
    
    def get_neighbour_angle_0(self):
        angle_0 = img_lib.get_angle(-self.neighbours_[0], \
            np.subtract(self.neighbours_[1], self.neighbours_[0]))
        return angle_0
    
    def get_neighbour_angle_1(self):
        angle_1 = img_lib.get_angle(-self.neighbours_[1], \
            np.subtract(self.neighbours_[0], self.neighbours_[1]))
        return angle_1

# the full set of triangle topogical features for a specified target ID
class topology_seq:
    def __init__(self, ID):
        self.base_ID_ = ID
        self.triangles_ = []
    
    def add_triangle(self, neighbours):
        triangle = delaunay_triangle(self.base_ID_, neighbours)
        self.triangles_.append(triangle)
    
    def topology_seq_size(self):
        return len(self.triangles_)
    
    def get_triangle(self, id):
        return self.triangles_[id]

    def get_triangle_base_angle(self, id):
        return self.triangles_[id].get_base_angle()
    
    def sum_of_base_angles(self):
        total = 0
        for triangle in self.triangles_:
            total += triangle.get_base_angle()
        return total

class camera(img_lib.camera_base):

    # extract data from csv files and generate other parameters
    def __init__(self, cam_ID, csv_file, img) -> None:
        super().__init__(cam_ID, csv_file, img)

        # topological data
        self.delaunay_map = []
        self.T = []
        for i in range(self.num_of_targets):
            t = topology_seq(i)
            self.T.append(t)
    
    # generate map of delaunay triangles for all targets in camera
    def generate_delaunay(self):
        self.delaunay_map = Delaunay(self.centroids)
    
    # generate 2d coords of all delaunay triangle vertices relative to a target
    def generate_2d_triangle_coords(self):
        for triangle in self.delaunay_map.simplices:
            for base in triangle:
                neighbours = []
                for vertex in triangle:
                    if not vertex == base:
                        dx = self.centroids[vertex][0] - self.centroids[base][0]
                        dy = self.centroids[vertex][1] - self.centroids[base][1]
                        neighbours.append(np.array([dx, dy]))
                self.T[base].add_triangle(neighbours)
    
    # generate 3d coords of all delaunay triangle vertices relative to a target
    # return coords of the 0th indexed triangle for 3d visualisation purposes
    def generate_3d_triangle_coords(self):

        x_data = [0]
        y_data = [0]
        z_data = [0]
        
        for triangle in self.delaunay_map.simplices:
            for base in triangle:
                base_area = self.areas[base]
                neighbours = []
                for vertex in triangle:
                    if not vertex == base:
                        dx = self.centroids[vertex][0] - self.centroids[base][0]
                        dy = self.centroids[vertex][1] - self.centroids[base][1]
                        area_ratio = self.areas[vertex]/base_area
                        dz = (1/area_ratio) - 1 # 3d path re-id, depth est formula
                        neighbours.append(np.array([dx, dy, dz]))
                self.T[base].add_triangle(neighbours)
                # 3d visualisation
                if base == 0:
                    for i in range(2):
                        x_data.append(neighbours[i][0])
                        y_data.append(neighbours[i][1])
                        z_data.append(neighbours[i][2])
        return x_data, y_data, z_data

#################
# re-id functions
#################

# source: Xudong et al (2022) https://www.mdpi.com/2504-446X/6/5/119 (Eq 4 - 6)

# w_iaib-x
def triangle_similarity(triangle_a, triangle_b):
    diff = 0
    diff += abs(triangle_a.get_base_angle() - triangle_b.get_base_angle())
    diff += abs(triangle_a.get_neighbour_angle_0() - triangle_b.get_neighbour_angle_0())
    diff += abs(triangle_a.get_neighbour_angle_1() - triangle_b.get_neighbour_angle_1())
    return 1 - math.log(1 + 1.72*diff/180)

# w_iaib
def get_weighted_similarity(topology_seq_a, topology_seq_b):

    # alpha_iaib
    denom = topology_seq_a.sum_of_base_angles() + topology_seq_b.sum_of_base_angles()

    # compare each triangle in a with each triangle in b
    numer = 0
    for i in range(topology_seq_a.topology_seq_size()):
        triangle_a = topology_seq_a.get_triangle(i)
        base_angle_a = topology_seq_a.get_triangle_base_angle(i)
        for j in range(topology_seq_b.topology_seq_size()):
            triangle_b = topology_seq_b.get_triangle(j)
            base_angle_b = topology_seq_b.get_triangle_base_angle(j)
            numer += (base_angle_a + base_angle_b) * \
                triangle_similarity(triangle_a, triangle_b)
    
    return numer / denom

###############
# main pipeline
###############

if parm.VISUALISE_3D:
    fig = plt.figure(figsize=plt.figaspect(0.5))
else:
    fig = plt.figure()
    ax = fig.add_subplot()

# extract info on detected objects
cameras = (camera(0, parm.csv_files[0], parm.img_files[0]), 
            camera(1, parm.csv_files[1], parm.img_files[1]))
for cam in cameras:
    cam.generate_delaunay()
    if parm.VISUALISE_3D:
        x_data, y_data, z_data = cam.generate_3d_triangle_coords()
        # 3d visualisation of delaunay triangles
        ax = fig.add_subplot(1, 2, cam.cam_ID+1, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if cam.cam_ID:
            m = 'o'
        else:
            m = '^'
        ax.scatter(x_data, y_data, z_data, marker=m)
    else:
        cam.generate_2d_triangle_coords()
        # 2d visualisation of delaunay triangles
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if cam.cam_ID:
            m = 'o'
        else:
            m = '^'
        ax.triplot(cam.centroids[:,0], cam.centroids[:,1], cam.delaunay_map.simplices)
        ax.scatter(cam.centroids[:,0], cam.centroids[:,1], marker=m)

# calculate similarity score matrix
W = np.zeros([cameras[0].num_of_targets, cameras[1].num_of_targets])
for a in cameras[0].IDs:
    for b in cameras[1].IDs:
        similarity = get_weighted_similarity(cameras[0].T[a], cameras[1].T[b])
        W[a,b] = similarity

# Re-ID visualisation
img_lib.annotate_and_save_reid(cameras[0].img, cameras[1].img,
                                cameras[0].centroids, cameras[1].centroids,
                                cameras[0].IDs, W, use_max=True)

plt.show()