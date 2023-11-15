from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("yolov8m.pt")

no_frame = trackid = bb_x_top_left = bb_y_top_left = bb_width = bb_height = 0
gender = 1
age = 2

# Open video file
CATEGORIES = os.listdir("Video_Phase1")
SUMISSION = ".\Submission"
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

        # Use YOLO to detect objects in the frame
        results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", verbose=False)
        # For each detected person...
        
        try:
            bboxs = results[0].boxes
            for i in range(len(bboxs.id)):
                trackid = int(bboxs.id[i])

                xywh = bboxs.xywh[i]

                # Get
                bb_x_top_left, bb_y_top_left, bb_width, bb_height = xywh

                res = f"{no_frame},{trackid},{bb_x_top_left:.2f},{bb_y_top_left:.2f},{bb_width:.2f},{bb_height:.2f},{gender},{age}"
                print(res)
                file_txt.write(res + "\n")
        except:
            continue

    file_txt.close()
    
    
if __name__ == '__main__':
    main()
    
