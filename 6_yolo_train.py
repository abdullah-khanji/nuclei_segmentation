import yaml
from ultralytics import YOLO  # Assuming you are using the ultralytics YOLO library
# import tensorflow as tf

# Load YOLO model
model = YOLO('yolov8n-seg.yaml')
model = YOLO('yolov8n-seg.pt')

# Load classes from data.yaml
with open('yolo_dataset/data.yaml') as f:
    classes = (yaml.safe_load(f)['nc'])

project = 'results/'
name = '10_epochs-'

# Train the model
results = model.train(
    data='yolo_dataset/data.yaml',
    name=name,
    project=project,
    epochs=10,
    patience=0,
    batch=16,
    imgsz=512
)
