import threading
import cv2
from ultralytics import YOLO
import os
from transformers import pipeline

pipe1 = pipeline("image-classification", model="NTQAI/pedestrian_age_recognition")
pipe2 = pipeline("image-classification", model="NTQAI/pedestrian_gender_recognition")

label = {
    "Age16-30": 2,
    "Age31-45": 3,
    "Age46-60": 4,
    "AgeAbove60": 5,
    "AgeLess15": 1,
    "Unknown": 0
}

def run_tracker_in_thread(filename, model, file_index):
    # ... (Your existing code for running the tracker in a thread)
    video = cv2.VideoCapture(filename)  # Read the video file
    no_frame = 0

    SUMISSION = ".\submission1"
    path = filename.replace('.mp4', '')
    path = path.replace('Video_Phase1\\','')
    print(path)
    save_path = os.path.join(SUMISSION,path)
    file_txt = open(save_path + ".txt","w")

    while True:
        # Read video frame
        no_frame += 1
        ret, frame = video.read()

        # Break if there are no frame no more
        if frame is None:
            break

        # Dictionary to check for old id
        id_dict = {}

        # Use YOLO to detect objects in the frame
        results = model.track(frame, persist=True, classes=0, tracker="bytetrack.yaml", verbose=False, show=False)
        
        # For each detected person...
        try:
            bboxs = results[0].boxes
            for i in range(len(bboxs.id)):
                trackid = int(bboxs.id[i])
                xywh = bboxs.xywh[i]
                bb_x_top_left, bb_y_top_left, bb_width, bb_height = xywh
                age = gender = 0

                if trackid not in id_dict:
                    # Get person in frame
                    person = frame[int(bboxs.xyxy[i][1]):int(bboxs.xyxy[i][3]), int(bboxs.xyxy[i][0]):int(bboxs.xyxy[i][2])]
                    
                    # Save img to a .jpg file
                    cv2.imwrite('person_crop.jpg',person)
                    
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

                # Get result
                res = f"{no_frame},{trackid},{bb_x_top_left:.2f},{bb_y_top_left:.2f},{bb_width:.2f},{bb_height:.2f},{gender},{age}"
                print(res)
                file_txt.write(res + "\n")

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        except:
            continue

    # Release video sources
    video.release()
    file_txt.close()

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Define the folder containing video files
video_folder = "Video_Phase1"

# Get a list of video files in the folder
video_files = [os.path.join(video_folder, filename) for filename in os.listdir(video_folder) if filename.endswith(".mp4")]

# Create a list to hold the tracker threads
tracker_threads = []

# Start a tracker thread for each video file
for i, video_file in enumerate(video_files):
    thread = threading.Thread(target=run_tracker_in_thread, args=(video_file, model, i + 1), daemon=True)
    tracker_threads.append(thread)

# Start all the tracker threads
for thread in tracker_threads:
    thread.start()

# Wait for all tracker threads to finish
for thread in tracker_threads:
    thread.join()

# Clean up and close windows
cv2.destroyAllWindows()
