from ultralytics import YOLO

model = YOLO("yolov8m-pose.pt")

results = model(source="test/ken_bad.jpg", show=False, conf=0.3, save=False, stream=True)

for result in results:
    keypoints = result.keypoints
    keypoints = results.keypoints.cpu().numpy().data

    # print(result.keypoints)
    print(keypoints)