import cv2
from ultralytics import YOLO
import numpy as np
import torch

# Load a model
# pretrained YOLOv8n model
model=YOLO('yolov8line.pt')
model.to('cuda')

#Function to load names file
def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

#Function to check for yellowline and platform edge crossing
def is_crossing_line(x0,y0,x1,y1,x2,y2):                
    #(x0,y0):bottom point of the line
    #(x1,y1):top point of the line
    #(x2,y2):third point to be checked
    val=((x1 - x0)*(y2 - y0)) - ((x2 - x0)*(y1 - y0))
    if val<=0:
        return True
    return False  # Line crossing detected for this person

# Example usage
class_names = load_class_names('coco.names')  # Change the file path accordingly
person_class_id = class_names.index('person')
yellow_class_id = class_names.index('yellow line')
edge_class_id = class_names.index('edge')

video_path=r"C:\Users\Vansh Patel\OneDrive\Desktop\Python\Video dataset\Jump_On_Track_1.avi"
cam = cv2.VideoCapture(video_path)

# Open webcam
font = cv2.FONT_HERSHEY_PLAIN
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cam.read()
results=model(frame)

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
            if confidence>0.5 and class_detect == "yellow line":
                line_x_bottom=x
                line_y_bottom=y+h
                line_x_top=x+w
                line_y_top=y
            if confidence>0.5 and class_detect == "edge":
                edge_x_bottom=x
                edge_y_bottom=y+h
                edge_x_top=x+w
                edge_y_top=y

#Checking the camera angle
if is_crossing_line(line_x_bottom,line_y_bottom,line_x_top,line_y_top,edge_x_bottom,edge_y_bottom):
    direction="L"
else:
    direction="R"

model=YOLO('yolov8m.pt')
model.to('cuda')

while True:
    ret, frame = cam.read()

    results=model(frame)
    
    for output in results:
        parameter=output.boxes
        for box in parameter:
            center_x,center_y,w,h=box.xywh[0]
            center_x,center_y,w,h=int(center_x),int(center_y),int(w),int(h)
            x = center_x - w // 2  # Top-left x-coordinate
            y = center_y - h // 2  # Top-left y-coordinate
            confidence=box.conf[0]
            class_detect=box.cls[0]
            class_detect=int(class_detect)
            class_detect=class_names[class_detect]
            if direction=="L":
                if confidence>0.5 and class_detect == "person" and is_crossing_line(edge_x_bottom,edge_y_bottom,edge_x_top,edge_y_top,center_x,y+h):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0,225), 2)
                    cv2.putText(frame, "!!!ON THE TRACK!!!", (x, y-10), font, 1, (0, 0,225), 2)
                elif confidence>0.5 and class_detect == "person" and is_crossing_line(line_x_bottom,line_y_bottom,line_x_top,line_y_top,center_x,y+h):  # Check crossing line for this person only
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 225), 2)
                    cv2.putText(frame, "Line Crossing!", (x, y-10), font, 1, (0, 225, 225), 2)
                elif confidence>0.5 and class_detect == "person":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (225,0,0), 2)
                    cv2.putText(frame, class_detect, (x, y - 5), font, 1, (225,0,0), 2)
            elif direction=="R":
                if confidence>0.5 and class_detect == "person" and ~is_crossing_line(edge_x_bottom,edge_y_bottom,edge_x_top,edge_y_top,center_x,y+h):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0,225), 2)
                    cv2.putText(frame, "!!!ON THE TRACK!!!", (x, y-10), font, 1, (0, 0,225), 2)
                elif confidence>0.5 and class_detect == "person" and ~is_crossing_line(line_x_bottom,line_y_bottom,line_x_top,line_y_top,center_x,y+h):  # Check crossing line for this person only
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 225), 2)
                    cv2.putText(frame, "Line Crossing!", (x, y-10), font, 1, (0, 225, 225), 2)
                elif confidence>0.5 and class_detect == "person":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (225,0,0), 2)
                    cv2.putText(frame, class_detect, (x, y - 5), font, 1, (225,0,0), 2)

    cv2.line(frame, (line_x_bottom,line_y_bottom), (line_x_top,line_y_top), (0,225,225), 3)
    cv2.line(frame, (edge_x_bottom,edge_y_bottom), (edge_x_top,edge_y_top), (0,225,225), 3)

    # Display output
    cv2.imshow("Video Analysis", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
