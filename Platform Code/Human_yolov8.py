import cv2
from ultralytics import YOLO
import numpy as np
import torch
model = YOLO('yolov8m.pt')
# Check if GPU is available
# if torch.cuda.is_available():
  # Move model to GPU (assuming your model is in a variable called 'model')
model.to('cuda')

def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

# Example usage
class_names = load_class_names('coco.names')  # Change the file path accordingly
person_class_id = class_names.index('person')


# Load a model
# pretrained YOLOv8n model
video_path=r"C:\Users\Vansh Patel\OneDrive\Desktop\Python\Video dataset\Passenger_Jump_2_2.mp4"
cam = cv2.VideoCapture(video_path)

# Process results list


def is_crossing_line(x,y):
    val=((894-497)*(y-1056))-((x-497)*(189-1056))
    if val<0:
        return True
    return False  # Line crossing detected for this person
# Open webcam
font = cv2.FONT_HERSHEY_PLAIN
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cam.read()

    results=model(frame)
    height, width,_ = frame.shape
    # Process detections
    boxes = []
    confidences = []
    class_ids = []
    
    for output in results:
        parameter=output.boxes
        for box in parameter:
            center_x,center_y,w,h=box.xywh[0]
            center_x,center_y,w,h=int(center_x),int(center_y),int(w),int(h)
            x = center_x - w // 2  # Top-left x-coordinate (center minus half width)
            y = center_y - h // 2  
            confidence=box.conf[0]
            class_detect=box.cls[0]
            class_detect=int(class_detect)
            class_detect=class_names[class_detect]
            if confidence>0.5 and class_detect == "person":
                cv2.rectangle(frame, (x, y), (x + w, y + h), (225,0,0), 2)
                cv2.putText(frame, class_detect, (x, y - 5), font, 1, (225,0,0), 2)
                if is_crossing_line(center_x,y+h):  # Check crossing line for this person only
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 2)  # Blue bounding box
                    cv2.putText(frame, "Line Crossing!", (x, y-10), font, 1, (0, 0, 225), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    thickness = 2  # Adjust thickness as needed
    line_color = (225,0,0)  # Yellow color (BGR)
    cv2.line(frame, (497,1056), (894,189), line_color, thickness)

    # Display output
    cv2.imshow("YOLO Object Detection (Persons only)", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
# import torch
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))