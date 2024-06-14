from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import yaml

model= YOLO('yolov8n-seg.yaml')
model= YOLO('yolov8n-seg.pt') #weights

with open('yolo_dataset/data.yaml') as stream:
    num_classes=str(yaml.safe_load(stream)['nc'])
    
    
project= '/results'

name="3_epochs-"


results= model.train(
    data= 'yolo_dataset/data.yaml',
    name=name,
    project=project,
    epochs=3, 
    patience=0,
    batch=4,
    imgsz=512
)
