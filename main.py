from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")  # load a pretrained model

# Train the model
results = model.train(data=r"C:\Users\Diya Shetty\OneDrive\Documents\Projects\Image_classification_Yolov8\data\weather_dataset",
                      epochs=20, imgsz=64)