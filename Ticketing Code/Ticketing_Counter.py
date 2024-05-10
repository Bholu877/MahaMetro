import cv2
from ultralytics import YOLO
import numpy as np
import torch

model = YOLO(r'Training_Data_counter\counter_detect.pt')

model1 = YOLO('yolov8l.pt')
def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def in_vision(x,y):
    result=cv2.pointPolygonTest(np.array(area,np.int32),((x,y)),False)
    if result>=0:
        return True
    else:
        return False

# Example usage
class_names = load_class_names('coco_ticketing.names')  # Change the file path accordingly
# person_class_id = class_names.index('queue line')

video_path=r"Final_Dataset\Ticketing_Line\Ticketing_Line_1_Cam1_1.avi"
cam = cv2.VideoCapture(video_path)

font = cv2.FONT_HERSHEY_PLAIN
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    # Resize the frame
    # frame = cv2.resize(frame, (640, 480))
    
    results = model(frame)
    
    height, width,_ = frame.shape
    detected_class=[]
    for output in results:
        parameter = output.boxes
        for box in parameter:
            center_x, center_y, w, h = box.xywh[0]
            center_x, center_y, w, h = int(center_x), int(center_y), int(w), int(h)
            x = center_x - w // 2  
            y = center_y - h // 2  
            # roi_x1 = x
            # roi_x2 = x+w
            # roi_y1 = y
            # roi_y2 = y+h
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = class_names[class_detect]
            if confidence > 0.5 and class_detect == "queues":
                area=[(x,y),(x,y+h),(x+w,y+h),(x+w,y),(x,y)]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (225,0,0), 2)
                cv2.putText(frame, class_detect, (x, y - 5), font, 1, (225,0,0), 2)
    results1 = model1(frame)
    count = 0
    for output1 in results1:
        parameter1 = output1.boxes
        for box1 in parameter1:
            center_x, center_y, w, h = box1.xywh[0]
            center_x, center_y, w, h = int(center_x), int(center_y), int(w), int(h)
            x = center_x - w // 2  
            y = center_y - h // 2  
            # roi_x1 = x
            # roi_x2 = x+w
            # roi_y1 = y
            # roi_y2 = y+h
            confidence = box1.conf[0]
            class_detect = box1.cls[0]
            class_detect = int(class_detect)
            class_detect = class_names[class_detect]
            if confidence > 0.5 and class_detect == "person" and in_vision(center_x,center_y):
                count +=1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame, class_detect, (x, y - 5), font, 1, (225,0,0), 2)
            if count > 10:
                print("More than 10 people in queue")
    print(count)
        # overlay = frame.copy()
        # cv2.polylines(overlay, pts = area_roi, isClosed = True, color=(255, 0, 0),thickness=2)
        # cv2.fillPoly(overlay, area_roi, (255,0,0))
        # frame = cv2.addWeighted(overlay, alpha,frame , 1 - alpha, 0)
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,0,225),2)
    cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Analysis", 1920, 1080) 
    cv2.imshow("Video Analysis", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()