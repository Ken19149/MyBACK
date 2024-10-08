from ultralytics import YOLO
import numpy as np
import cv2  # OpenCV for drawing
import matplotlib.pyplot as plt

model = YOLO("yolov8n-pose.pt")

image_path = "test/ken_bad.jpg"
results = model(source=image_path, show=True, conf=0.3, save=True)

image = cv2.imread(image_path)

# Iterate over each detected person in the results
for result in results:
    if hasattr(result, 'keypoints'):
        # Each result corresponds to one detected object/person
        for keypoint_set in result.keypoints:  # Loop through each person's keypoints
            # Extract keypoint coordinates from the data tensor
            keypoints = result.keypoints.cpu().numpy().data # Convert to NumPy array
            print("Extracted Keypoints:", keypoints)  # Debug print

            # List to store lengths between consecutive keypoints
            lengths = []

            # Calculate the length between consecutive keypoints and draw on the image
            for i in range(len(keypoints) - 1):
                x1, y1 = keypoints[i][:2]  # Only take (x, y) ignoring other dimensions
                x2, y2 = keypoints[i + 1][:2]

                # Euclidean distance between two keypoints
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                lengths.append(length)

                # Draw a line between the keypoints
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green line

                # Calculate the midpoint to place the text
                midpoint_x = int((x1 + x2) / 2)
                midpoint_y = int((y1 + y2) / 2)

                # Draw the length near the midpoint
                cv2.putText(image, f"{length:.2f}", (midpoint_x, midpoint_y), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 0, 0), 2)  # Blue text

# Save the image with annotations
output_path = "output_with_lengths.jpg"  # Change to your desired output path
cv2.imwrite(output_path, image)

# Display the image using Matplotlib (if environment supports it)
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axes
plt.show()