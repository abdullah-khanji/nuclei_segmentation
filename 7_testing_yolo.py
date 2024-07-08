from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
model_path= 'results/10_epochs-6/weights/last.pt'
model= YOLO(model_path)

img= 'test1_liver.png'

result= model.predict(img, conf=0.5)
print(result[0])
result_array= result[0].plot()
plt.figure(figsize=(9, 9))
plt.imshow(result_array)
plt.show()