import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Images
imgs = 'data/pics/5drones_0.png'

# Inference
results = model(imgs)
# results.print()
# results.save()
results.show()
print('\n', results.pandas().xyxy[0])