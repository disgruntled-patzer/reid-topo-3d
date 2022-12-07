# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html

import csv
import cv2
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
img_files = [
    'data/pics/frisbee_0.png',
    'data/pics/frisbee_1.png'
]
colours = [
    (0, 255, 0), # 1
    (255, 0, 0), # 2
    (0, 0, 255), # 3
    (143, 89, 255), # 4
    (6, 39, 156), # 5
    (92, 215, 206), # 6
    (105, 139, 246), # 7
    (84, 43, 0), # 8
    (137, 171, 197), # 9
    (147, 226, 255) # 10
]

# extract data from csv file
def extract_data(src):
    extracted = []
    with open (src, 'r') as csvfile:
        csvreader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in csvreader:
            extracted.append(row)
    return extracted

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

class camera:

    # extract data from csv files and generate other parameters
    def __init__(self, cam_ID, csv_file, img) -> None:

        # csv data
        self.cam_ID = cam_ID
        self.csv_data = extract_data(csv_file)
        self.img = cv2.imread(img)

        # basic target data. IDs are zero indexed
        self.num_of_targets = len(self.csv_data)
        self.IDs = list(range(0,self.num_of_targets))
        self.areas = np.ones(self.num_of_targets)
        self.centroids = np.ones([self.num_of_targets,2])

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

cameras = (camera(0, csv_files[0], img_files[0]), camera(1, csv_files[1], img_files[1]))
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

# to-do: Display Re-ID results
print(assoc_err)

# "matched" - for each target ID in Camera 0 (represented by the row of "matched"), 
# get the ID of the matched target in Camera 1 (represented by the value in "matched")
matched = np.argmin(assoc_err, axis=1)

# annotate re-id images
combined_img = cv2.hconcat([cameras[0].img, cameras[1].img])

for id0 in cameras[0].IDs:
    # Camera 0 "id0" is matched with Camera 1 "id1"
    id1 = matched[id0]

    # generate target centroid coordinates to be annotated on the frame
    id0_coords = cameras[0].centroids[id0].copy()
    # transform from standard coords (origin at bottom left) to img coords (origin at top left)
    id0_coords[1] = IMG_HEIGHT - id0_coords[1]
    # convert centroid floats to int
    id0_coords = [int(x) for x in id0_coords]
    # convert to tuple to be compatible with OpenCV line
    id0_coords = tuple(id0_coords)

    id1_coords = cameras[1].centroids[id1].copy()
    id1_coords[1] = IMG_HEIGHT - id1_coords[1]
    # transpose to the right for Camera 1 IDs
    id1_coords[0] += IMG_WIDTH
    id1_coords = [int(x) for x in id1_coords]
    id1_coords = tuple(id1_coords)

    combined_img = cv2.line(combined_img, id0_coords, id1_coords, colours[id0%10], 3)

cv2.imwrite('data/pics/saved.jpg', combined_img)

plt.show()