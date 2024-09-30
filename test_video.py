import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (v8n is the nano version, ideal for edge devices like Raspberry Pi)
model = YOLO('yolov8n.pt')  # Replace with the path if necessary

# Load your video file
video_path = 'video/floor_5.mp4'  # Replace with your actual video file path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no frame is returned

    # resize frame
    frame = cv2.resize(frame, (640, 480))

    # Perform YOLOv8 object detection
    results = model(frame)

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()  # Draw bounding boxes and labels on the frame

    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()