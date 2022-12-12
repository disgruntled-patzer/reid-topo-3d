# reid_delaunay_rot.py
# Lau Yan Han (2022)
#
# Overview:
# Similar to reid_delaunay.py. However, the 3D coordinates of
# all targets are estimated relative to the 1st detected target
# and then Euler rotation is performed to transform these coordinates
# into a global frame. Then the re-identification process and output 
# is similar to reid_delaunay.py.
#
# Inputs: 
# 1. csv files with detected targets (one for each target). Stored as
# a size 2 list in variable csv_files, one cell for each camera
# 2. original image files. Stored as a size 2 list in variable img_files,
# one cell for each camera. Use for visualisation purposes later
# 3. "EULER_*" set of parameters. These specify the Euler Angles
# (in degrees) of each camera relative to the global frame
# 
# Output: visualisation of re-id results across both images - saved in
# 'data/pics' folder and displayed in a new window
# 
# Convention for global frame:
# The global frame is similar to a camera pointing vertically 
# downwards at the plane containing the targets, with Camera 0's 
# y-axis aligned with the global frame.

import img_lib
import img_parameters as parm
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Delaunay

#############################
# camera and topology classes
#############################

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
    def __init__(self, cam_ID, csv_file, img, eul_ang) -> None:
        super().__init__(cam_ID, csv_file, img)

        # transformed target data
        self.coords3d = np.zeros([self.num_of_targets,3])
        self.transformed = np.zeros([self.num_of_targets,3])

        # Euler ZYX angle rotation matrix from global frame to camera
        eul_x = math.radians(eul_ang[0])
        eul_y = math.radians(eul_ang[1])
        eul_z = math.radians(eul_ang[2])
        self.rot_matrix = np.zeros([3,3])
        self.rot_matrix[0][0] = math.cos(eul_y)*math.cos(eul_z)
        self.rot_matrix[0][1] = -math.cos(eul_x)*math.sin(eul_z) + math.sin(eul_x)*math.sin(eul_y)*math.cos(eul_z)
        self.rot_matrix[0][2] = math.sin(eul_x)*math.sin(eul_z) + math.cos(eul_x)*math.sin(eul_y)*math.cos(eul_z)
        self.rot_matrix[1][0] = math.cos(eul_y)*math.sin(eul_z)
        self.rot_matrix[1][1] = math.cos(eul_x)*math.cos(eul_z) + math.sin(eul_x)*math.sin(eul_y)*math.sin(eul_z)
        self.rot_matrix[1][2] = -math.sin(eul_x)*math.cos(eul_z) + math.cos(eul_x)*math.sin(eul_y)*math.sin(eul_z)
        self.rot_matrix[2][0] = -math.sin(eul_y)
        self.rot_matrix[2][1] = math.sin(eul_x)*math.cos(eul_y)
        self.rot_matrix[2][2] = math.cos(eul_x)*math.cos(eul_y)

        # we will rotate from camera back to global frame, thus the rotation matrix must be inversed
        self.rot_matrix = np.linalg.inv(self.rot_matrix)

        # topological data
        self.delaunay_map = []
        self.T = []
        for i in range(self.num_of_targets):
            t = topology_seq(i)
            self.T.append(t)
    
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
    
    # transform given 3d coordinates from camera to global frame
    def transform_coords(self):
        for id in range(self.num_of_targets):
            self.transformed[id,:] = np.matmul(self.rot_matrix, self.coords3d[id,:])
    
    # generate map of delaunay triangles for 2D XY projections of transformed targets
    def generate_delaunay(self):
        projections_2d = self.transformed[:,:2]
        self.delaunay_map = Delaunay(projections_2d)
        # print(self.delaunay_map.simplices[0][0])
        # simplices -> triangle (1st array id) -> vertex (2nd array id)
    
    # generate 2d vectors of all delaunay triangle vertices relative to a target
    def generate_delaunay_triangle_vectors(self):
        for triangle in self.delaunay_map.simplices:
            for base in triangle:
                neighbours = []
                for vertex in triangle:
                    if not vertex == base:
                        dx = self.centroids[vertex][0] - self.centroids[base][0]
                        dy = self.centroids[vertex][1] - self.centroids[base][1]
                        neighbours.append(np.array([dx, dy]))
                self.T[base].add_triangle(neighbours)

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
cameras = (
    camera(0, parm.csv_files[0], parm.img_files[0], [parm.EULER_X_0, parm.EULER_Y_0, parm.EULER_Z_0]), 
    camera(1, parm.csv_files[1], parm.img_files[1], [parm.EULER_X_1, parm.EULER_Y_1, parm.EULER_Z_1])
    )
for cam in cameras:
    # transformation and topography map generation
    cam.generate_3d_coords()
    cam.transform_coords()
    cam.generate_delaunay()
    cam.generate_delaunay_triangle_vectors()

    # 3d visualisation
    if parm.VISUALISE_3D:
        ax = fig.add_subplot(1, 2, cam.cam_ID+1, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if cam.cam_ID:
            m = 'o'
        else:
            m = '^'
        ax.scatter(cam.transformed[:,0], cam.transformed[:,1], cam.transformed[:,2], marker=m)
    else:
        # 2d visualisation
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if cam.cam_ID:
            m = 'o'
        else:
            m = '^'
        ax.triplot(cam.transformed[:,0], cam.transformed[:,1], cam.delaunay_map.simplices)
        ax.scatter(cam.transformed[:,0], cam.transformed[:,1], marker=m)

# calculate similarity score matrix W
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