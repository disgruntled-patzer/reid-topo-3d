import csv
# import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Delaunay

# data format saved by Yolo V5:
# id    xmin    ymin    xmax   ymax  confidence  class    name
#  0  749.50   43.50  1148.0  704.5    0.874023      0  person
#  1  433.50  433.50   517.5  714.5    0.687988     27     tie
#  2  114.75  195.75  1095.0  708.0    0.624512      0  person
#  3  986.00  304.00  1028.0  420.0    0.286865     27     tie
# (xmin,ymin) is top-left, (xmax,ymax) is lower right. (0,0) is top-left of img

##########################
# functions and parameters
##########################

VISUALISE_3D = False
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
XMIN = 1
YMIN = 2
XMAX = 3
YMAX = 4
csv_files = [
    'data/csv/ball_3.csv',
    'data/csv/ball_4.csv'
]

# Euler angles (deg) and position of Camera 1 w.r.t. Camera 0
EULER_Z = 0.0
EULER_Y = -30.0
EULER_X = 0.0
POS_Z = 0.0
POS_Y = 0.0
POS_X = 0.0

# get angle (radian) between two vectors a and b (represented as arrays)
def get_angle(a, b):
    dot_product = np.dot(a, b)
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    return np.arccos(dot_product / norm_product)

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

class delaunay_triangle:
    def __init__(self, ID, neighbours):
        self.base_ID_ = ID
        self.neighbours_ = neighbours
    
    def get_base_ID(self):
        return self.base_ID_
    
    def get_base_angle(self):
        return get_angle(self.neighbours_[0], self.neighbours_[1])
    
    def get_neighbour_angle_0(self):
        angle_0 = get_angle(-self.neighbours_[0], \
            np.subtract(self.neighbours_[1], self.neighbours_[0]))
        return angle_0
    
    def get_neighbour_angle_1(self):
        angle_1 = get_angle(-self.neighbours_[1], \
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

class camera:

    # extract data from csv files and generate other parameters
    def __init__(self, cam_ID, csv_file, eul_ang, rel_pos) -> None:

        # csv data
        self.cam_ID = cam_ID
        self.csv_data = extract_data(csv_file)

        # basic target data. IDs are zero indexed
        self.num_of_targets = len(self.csv_data)
        self.IDs = list(range(0,self.num_of_targets))
        self.areas = np.ones(self.num_of_targets)
        self.centroids = np.ones([self.num_of_targets,2])

        # transformed target data
        self.coords3d = np.zeros([self.num_of_targets,3])
        self.transformed = np.zeros([self.num_of_targets,3])

        # euler (ZYX) rotation matrix and position vector relative to global frame
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
        self.pos_vec = np.array(rel_pos)

        # topological data
        self.delaunay_map = []
        self.T = []
        for i in range(self.num_of_targets):
            t = topology_seq(i)
            self.T.append(t)
    
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
    
    # transform given 3d coordinates from camera to global frame
    def transform_coords(self):
        for id in range(self.num_of_targets):
            self.transformed[id,:] = self.pos_vec + np.matmul(self.rot_matrix, self.coords3d[id,:])
    
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

if VISUALISE_3D:
    fig = plt.figure(figsize=plt.figaspect(0.5))
else:
    fig = plt.figure()
    ax = fig.add_subplot()

# extract info on detected objects
cameras = (
    camera(0, csv_files[0], [0.0,0.0,0.0], [0.0,0.0,0.0]), 
    camera(1, csv_files[1], [EULER_X, EULER_Y, EULER_Z], [POS_X, POS_Y, POS_Z])
    )
for cam in cameras:
    cam.get_target_areas_centroids()
    cam.generate_3d_coords()
    cam.transform_coords()
    cam.generate_delaunay()
    cam.generate_delaunay_triangle_vectors()

    # 3d visualisation
    if VISUALISE_3D:
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

# calculate similarity score matrix
W = np.zeros([cameras[0].num_of_targets, cameras[1].num_of_targets])
for a in cameras[0].IDs:
    for b in cameras[1].IDs:
        similarity = get_weighted_similarity(cameras[0].T[a], cameras[1].T[b])
        W[a,b] = similarity
print(W)

plt.show()