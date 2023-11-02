from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
import time
from transformers import pipeline

pipe1 = pipeline("image-classification", model="NTQAI/pedestrian_age_recognition")
pipe2 = pipeline("image-classification", model="NTQAI/pedestrian_gender_recognition")
# Load YOLO model
model = YOLO("yolov8m.pt")

no_frame = trackid = bb_x_top_left = bb_y_top_left = bb_width = bb_height = gender = 0
age = 2
# Open video file
CATEGORIES = os.listdir("Video_Phase1")
SUMISSION = ".\submission2"

label = {
    "Age16-30": 2,
    "Age31-45": 3,
    "Age46-60": 4,
    "AgeAbove60": 5,
    "AgeLess15": 1,
    "Unknown": 0
}

def main():
    for video in CATEGORIES:
        print(video)
        path = video
        video = cv2.VideoCapture("Video_Phase1/" + path)
        path = path.replace('.mp4', '')
        save_path = os.path.join(SUMISSION,path)
        f = open(save_path + ".txt", "w")
        dectect(video,f)
        


def dectect(video, file_txt):
    no_frame = 0
    print(video)
    print(file_txt)
    while True:
        # Read video frame
        no_frame += 1
        ret, frame = video.read()

        # Break if there are no frame no more
        if frame is None:
            break
        id_dict = {}
        # Use YOLO to detect objects in the frame
        results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", verbose=False)
        # For each detected person...
        try:
            bboxs = results[0].boxes
            for i in range(len(bboxs.id)):
                trackid = int(bboxs.id[i])
                xywh = bboxs.xywh[i]
                bb_x_top_left, bb_y_top_left, bb_width, bb_height = xywh
                age = gender = 0

                if trackid not in id_dict:

                    person = frame[int(bboxs.xyxy[i][1]):int(bboxs.xyxy[i][3]), int(bboxs.xyxy[i][0]):int(bboxs.xyxy[i][2])]
                    
                    img_path = cv2.imwrite('person_crop.jpg',person)
                    
                    age = pipe1('person_crop.jpg')
                    gender = pipe2('person_crop.jpg')

                    age_score = age[0]['score']
                    age_label = None
                    if age_score>0.65:
                        age_label = age[0]['label']
                    else:
                        age_label = "Unknown"

                    age = label[age_label]

                    gender_label = gender[0]['label']
                    if gender_label == "Male":
                        gender = 0
                    else:
                        gender = 1
                    id_dict[trackid] = (age,gender)



                age, gender = id_dict[trackid]
                res = f"{no_frame},{trackid},{bb_x_top_left:.2f},{bb_y_top_left:.2f},{bb_width:.2f},{bb_height:.2f},{gender},{age}"
                print(res)
                file_txt.write(res + "\n")
        except:
            continue
        

    file_txt.close()
    
    
if __name__ == '__main__':
    main()