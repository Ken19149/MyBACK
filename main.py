from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")

results = model(source="test/1.jpg", show=False, conf=0.3, save=False, stream=False)


for result in results:
    keypoints = result.keypoints
    print(result.keypoints.xy)


