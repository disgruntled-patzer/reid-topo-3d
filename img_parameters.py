# img_parameters.py
# Lau Yan Han (2022)
#
# Parameters file for all image/topology based Re-ID methods
# Use this library for Re-ID algorithms that operate on instantaneous
# snapshots of detected targets.

##############################
# data format saved by Yolo V5
##############################

# id    xmin    ymin    xmax   ymax  confidence  class    name
#  0  749.50   43.50  1148.0  704.5    0.874023      0  person
#  1  433.50  433.50   517.5  714.5    0.687988     27     tie
#  2  114.75  195.75  1095.0  708.0    0.624512      0  person
#  3  986.00  304.00  1028.0  420.0    0.286865     27     tie
# (xmin,ymin) is top-left, (xmax,ymax) is lower right. (0,0) is top-left of img

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