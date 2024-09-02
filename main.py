from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")

result = model(source=0, show=True, conf=0.3, save=True)