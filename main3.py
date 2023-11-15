from ultralytics import YOLO
import cv2
import numpy as np
import torch

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video file
path = "Kabu_88.mp4"
video = cv2.VideoCapture("Video_Phase1/" + path)
path = path.replace('.mp4', '')
f = open(path + ".txt", "w")

# <no_frame>,<trackid>,<bb_x_top_left>,<bb_y_top_left>,<bb_width>,<bb_height>,<age>,<gender>
no_frame = trackid = bb_x_top_left = bb_y_top_left = bb_width = bb_height = gender = 0
age = 2

while True:
    # Read video frame
    no_frame += 1
    ret, frame = video.read()

    # Break if there are no frame no more
    if frame is None:
        break

    # Use YOLO to detect objects in the frame
    results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", verbose=False)
    # For each detected person...
    bboxs = results[0].boxes

    for i in range(len(bboxs.id)):
        trackid = int(bboxs.id[i])

        xywh = bboxs.xywh[i]

        # Get
        bb_x_top_left, bb_y_top_left, bb_width, bb_height = xywh

        res = f"{no_frame},{trackid},{bb_x_top_left:.2f},{bb_y_top_left:.2f},{bb_width:.2f},{bb_height:.2f},{gender},{age}"
        print(res)
        f.write(res + "\n")

f.close()
