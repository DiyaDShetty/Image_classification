import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO(r"C:\Users\Diya Shetty\runs\classify\train2\weights\last.pt")  # load a custom model

img_path=r"C:\Users\Diya Shetty\OneDrive\Documents\Projects\Image_classification_Yolov8\code\dataset\sunrise1.jpg"
img=cv2.imread(img_path)

results = model(img_path)  # predict on an image

names_dict=results[0].names

probs = results[0].probs.data.cpu().numpy().tolist()

print(names_dict)
print(probs)

print(names_dict[np.argmax(probs)])

cv2.imshow(names_dict[np.argmax(probs)], img)
cv2.waitKey(0)