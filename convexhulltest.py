# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html

import csv
import matplotlib.pyplot as plt
import numpy as np
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

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
XMIN = 1
YMIN = 2
XMAX = 3
YMAX = 4
csv_files = [
    'data/csv/frisbee_0.csv',
    'data/csv/frisbee_1.csv'
]

# extract data from csv file
def extract_data(src):
    extracted = []
    with open (src, 'r') as csvfile:
        csvreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in csvreader:
            extracted.append(row)
    return extracted

# return intersection of two 2d lines in x-y coordinates
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
        return x,y
    else:
        # lines are parallel if determinant = 0
        return None, None

# return cross ratio of 4 collinear 2D points (represented as arrays)
def cross_ratio(a, b, c, d):
    ac = np.linalg.norm(a,c)
    bd = np.linalg.norm(b,d)
    bc = np.linalg.norm(b,c)
    ad = np.linalg.norm(a,d)
    return (ac*bd)/(bc*ad)

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

        # topological data
        self.hull = []
    
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
    
    def generate_convexhull(self):
        self.hull = ConvexHull(self.centroids)
        if len(self.hull.vertices) < 5:
            raise Exception("Algorithm needs 5 or more convex hull vertices")
        print(self.hull.simplices)
        print(self.hull.vertices)
    
    # Given a convex hull vertex's position, get the target ID of the vertex and its 4 neighbours
    def get_vertice_and_neighbour_IDs(self, hull_pos):
        hull_nvertices = len(self.hull.vertices)
        if hull_pos >= hull_nvertices:
            raise Exception("Position exceeds number of convex hull vertices")
        positions = np.array([hull_pos - 2, hull_pos - 1, hull_pos, hull_pos + 1, hull_pos + 2])
        # wraparound the number of convex hull vertices
        positions = np.mod(positions, hull_nvertices)
        target_IDs = np.zeros(5, dtype=int)
        for i in range(5):
            target_IDs[i] = self.hull.vertices[positions[i]]
        return target_IDs

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('X')
ax.set_ylabel('Y')

cameras = (camera(0, csv_files[0]), camera(1, csv_files[1]))
for cam in cameras:
    cam.get_target_areas_centroids()
    cam.generate_convexhull()
    if cam.cam_ID:
        m = 'o'
    else:
        m = '^'
    plt.scatter(cam.centroids[:,0], cam.centroids[:,1], marker=m)
    for simplex in cam.hull.simplices:
        plt.plot(cam.centroids[simplex,0], cam.centroids[simplex,1], 'k-')

plt.show()