# detectobjects.py
# Lau Yan Han (2022)
#
# Detect objects using Yolo V5 for a given image. Specify
# the path to image in "imgs" variable. The program will
# output the detected info and coordinates on the terminal,
# and save/display the annotated images

import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
imgs = 'data/pics/5drones_0.png'

# Inference
results = model(imgs)
results.print()
# results.save()
results.show()
print('\n', results.pandas().xyxy[0])