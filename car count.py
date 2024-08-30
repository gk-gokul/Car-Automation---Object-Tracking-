import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import *

yolo_model = YOLO('yolov8s.pt')


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_position = [x, y]
        print(cursor_position)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', mouse_callback)

video_path = 'C:\\Users\\ecea1\\Desktop\\yolov8counting-trackingvehicles-main\\yolov8counting-trackingvehicles-main\\veh2.mp4'
video_capture = cv2.VideoCapture(video_path)

class_list_file_path = "C:\\Users\\ecea1\\Desktop\\yolov8counting-trackingvehicles-main\\yolov8counting-trackingvehicles-main\\coco.txt"
with open(class_list_file_path, "r") as class_list_file:
    class_data = class_list_file.read()
class_list = class_data.split("\n")

frame_count = 0
tracker = Tracker()

cy1 = 322
cy2 = 368
offset = 6

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    yolo_results = yolo_model.predict(frame)
    bbox_data = yolo_results[0].boxes.data
    bbox_df = pd.DataFrame(bbox_data).astype("float")

    car_bboxes = []

    for index, row in bbox_df.iterrows:
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        class_id = int(row[5])
        class_name = class_list[class_id]
        if 'car' in class_name:
            car_bboxes.append([x1, y1, x2, y2])

    tracked_car_ids = tracker.update(car_bboxes)

    for car_bbox in tracked_car_ids:
        x3, y3, x4, y4, car_id = car_bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, str(car_id), (cx, cy),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
