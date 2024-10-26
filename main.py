import cv2
from ultralytics import YOLO

# Load the YOLOv8 nano model (trained for object detection)
model = YOLO('yolov8n.pt')  # yolov8n.pt is the YOLOv8 nano model

# Define the video path
video_path = '240520/240520_055427_055527.mp4'
cap = cv2.VideoCapture(video_path)

# Output video settings (optional)
output_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 model on the frame
    results = model(frame)

    # Draw detections on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            label = result.names[int(box.cls[0])]  # Class label

            # Filter to only display trash cans (assuming class "trash can" is trained)
            if label == 'boat' and confidence > 0.3:  # adjust confidence threshold if needed
                label = 'trash can'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Trash Can Detection", frame)
    out.write(frame)  # Write to output video

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

