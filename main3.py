from ultralytics import YOLO
import cv2
import numpy

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load OpenCV models for face detection and age/gender prediction
faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Define age and gender categories
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Open video file
path = "Doto_103.mp4"
video = cv2.VideoCapture('Video_Phase1/'+path)
f = open(path+".txt", "w")

frame_count = 0

gender = 0
age = 2
while True:
    # Read video frame
    frame_count += 1
    ret, frame = video.read()

    # Use YOLO to detect objects in the frame
    results = model.track(frame,persist=True, classes=0, tracker="bytetrack.yaml")
    # For each detected person...
    bboxs = results[0].boxes
    for i in range(len(bboxs.id)):
        id= int(bboxs.id[i])
        x,y,w,h = bboxs.xywh[i]
        res = f"{frame_count},{id},{numpy.round(x,2)},{numpy.round(y,2)},{numpy.round(w,2)},{numpy.round(h,2)},{age},{gender}"
        print(res)
        f.write(res+"\n")
    if frame is None:
        break
f.close()       
    
    
    
    # for result in results:
    #     if result.names[0] == 'person':
            
            
            # Crop frame to detected person
            # for box in result.boxes:
            #     print(f"box: {box}")
            #     x1, y1, x2, y2 = box
                # height, width = frame.shape[:2]
                # x1 = int(x1 * width)
                # y1 = int(y1 * height)
                # x2 = int(x2 * width)
                # y2 = int(y2 * height)

                # person = frame[y1:y2, x1:x2]
                # if frame is None:
                #     print("Failed to load image or video frame")
                # else:
                #     person = frame[y1:y2, x1:x2]

                # # Use OpenCV to detect face in the cropped image
                # blob = cv2.dnn.blobFromImage(person, 1.0, (300, 300), [104, 117, 123], True, False)
                # faceNet.setInput(blob)
                # detections = faceNet.forward()

                # # For each detected face...
                # for i in range(detections.shape[2]):
                #     confidence = detections[0, 0, i, 2]
                #     if confidence > 0.7:
                #         # Predict age and gender of the face
                #         face = person[max(0, y1-15): min(y2+15, person.shape[0]-1), max(0, x1-15): min(x2+15, person.shape[1]-1)]
                #         if face is not None and face.size > 0:
                #             blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                #         else:
                #             print("No face detected or face image is empty")
                        
                #         genderNet.setInput(blob)
                #         genderPreds=genderNet.forward()
                #         gender=genderList[genderPreds[0].argmax()]
                        
                #         ageNet.setInput(blob)
                #         agePreds=ageNet.forward()
                #         age=ageList[agePreds[0].argmax()]
                        
                #         # Print or store the predictions as needed
                #         print(f"Gender: {gender}, Age: {age}")
