# General
import json
import serial
import time
from datetime import datetime
from queue import Queue, Full
import threading
import cv2
import joblib
import numpy as np
import os
# Pose
import torch
from ultralytics import YOLO
# Face Recogntion
import face_recognition
# Lidar
import rospy
from sensor_msgs.msg import LaserScan



print('....loading')


py_serial = serial.Serial(
        port='/dev/ttyACM0',
        baudrate=9600,)
db_serial = serial.Serial(
        port='/dev/ttyGS0',
        baudrate=9600)



input_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
video_queue = Queue(maxsize=1)
lidar_queue = Queue(maxsize=2)
face_queue = Queue(maxsize=1)
display_queue = Queue(maxsize=1)

images = []
face_encodings = []
image_paths = [os.path.join('./img', i) for i in os.listdir('./img')]
for image_path in image_paths:
    image = face_recognition.load_image_file(image_path)
    images.append(image)
    face_encodings.append(face_recognition.face_encodings(image)[0])

known_face_encodings = face_encodings
known_face_names = [i.split('.')[0] for i in os.listdir('./img')]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'----------{device}----------')


# rplidar a1 : sampling from -pi to pi with 1147 points
theta = np.linspace(-np.pi, np.pi, 1147) 

# Lidar
def lidar_callback(data):

    ldr = data.ranges


    # Cylindrical coordinate To Cartesian coordinate
    x = ldr * np.cos(theta)
    y = ldr * np.sin(theta)

    points_fps_x = []
    for i in range(len(x)):
        if (x[i] > -5.2) & (x[i] < -0.5) & (y[i] > -0.3) & (y[i] < 0.3):
            points_fps_x.append(x[i])
        else:
            pass
    
    ldr_mean = np.array(points_fps_x).mean()
    lidar_queue.put(ldr_mean)

    # lidar_queue.put(points_fps_x)
    # print(points_fps_x)
    # lidar_queue.put(ldr)

def extract_skeletons(results):
    kp = results[0].keypoints.xyn
    if kp.cpu().numpy().size == 0:
        skeletons = torch.zeros(6)
    else:
        skeletons = torch.tensor([
                    torch.dist(kp[0][0], kp[0][11]).item(),
                    torch.dist(kp[0][0], kp[0][12]).item(),
                    torch.dist(kp[0][5], kp[0][6]).item(),
                    torch.dist(kp[0][5], kp[0][11]).item(),
                    torch.dist(kp[0][6], kp[0][12]).item(),
                    torch.dist(kp[0][11], kp[0][12]).item()])
    return skeletons

def capture_thread(input_queue, video_queue):
    cap = cv2.VideoCapture(0)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w / 3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, h / 3)
    while True:
        ret, image = cap.read()
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if ret:
            try:
                input_queue.put(image, block=False)
                video_queue.put(image, block=False)
            except Full:
                pass

def inference_thread(input_queue, output_queue):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO('yolov8n-pose.pt')
    model.to(device)
    
    while True:
        image = input_queue.get()

        #Pose Inference
        results = model.predict(image, conf=0.5)
        results[0] = results[0].to(device)

        try:
            output_queue.put_nowait(results)
        except:
            output_queue.get_nowait()
            output_queue.put_nowait(results)
        # output_queue.put(results)

        #Face Inference
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        
        #print("Process this frame: ", process_this_frame)

        if process_this_frame:
            #rgb_frame = image[:, :, ::-1]
            code = cv2.COLOR_BGR2RGB
            rgb_frame = cv2.cvtColor(image, code)

            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

            process_this_frame = not process_this_frame

            try:
                face_queue.put_nowait(face_names)
            except:
                face_queue.get_nowait()
                face_queue.put_nowait(face_names)
            # if len(face_names) != 0:
            #     face_queue.put_nowait(face_names)




def classification_thread(output_queue, lidar_queue, video_queue, face_queue):
    # initalize classifier
    model = joblib.load('knn_model_realtime.pkl')
    seq = []
    seq_num = 10 # prediction with 10 sequence
    anomal_ratio = 0 
    
    # Video writer setup
    file_path = f'results/record_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.avi'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    
    size = (480,640) 
    out = cv2.VideoWriter(file_path, fourcc, 5.0, size)  # �뚯씪紐�, 肄붾뜳, �꾨젅�� ��, �댁긽��
    while True:
        results = output_queue.get()
        pose_img = results[0].plot()

        results_lidar = lidar_queue.get()
        print('results_lidar',results_lidar)
        skeletons = extract_skeletons(results)  # Length between Landmarks
        
        # Optional: Run pose coordinates through RandomForest Classifier
        seq.append(skeletons/results_lidar)
        if len(seq) == seq_num:
            seq = np.stack(seq)
            distances, indices = model.kneighbors(seq)
            threshold = 0.0001
            #print(f"INDICES : {indices}")
            anomalies = np.where(np.mean(distances,axis=1) > threshold)[0]
            anomal_ratio = len(anomalies) / seq_num
            seq = [] # sequence reset
            
            print('len anomalies', len(anomalies))
        

            if anomal_ratio > 0.8:                        # Stranger
                try:
                    name = face_queue.get_nowait()
                        
                except queue.Empty:
                    print("========== FACE QUEUE IS EMPTY ===========")
                    return

                print(f'STRANGER!!! {name} ----STRANGER!!! {name}----STRANGER!!! {name}----STRANGER!!! {name}----STRANGER!!!----STRANGER!!!')
                print(f'anomal_ratio : {anomal_ratio}')
                
                log = {
                        'type': 'Stranger',
                        'joint': skeletons.tolist(),
                        'anomal': anomal_ratio,
                        # 'timestamp' : datetime.now() 
                        }
            else:
                try:
                    name = face_queue.get_nowait()
                except queue.Empty:
                    print("QUEUE IS EMPTY")
                    return
                
                log = {
                        'type': 'User',
                        'joint': skeletons.tolist(),
                        'anomal': anomal_ratio,
                        # 'timestamp': datetime.now()
                        }
                if len(name) == 0:                        # 2차인증 (얼굴인식)
                        print("Pass 1st Authentication but Cannot Pass 2nd authentication. Keep the Door Locked.")
                        print(f'anomal_ratio : {anomal_ratio}')
                else:
                        print(f'---------Welcome {name} ------------------Welcome {name}------------------Welcome {name}---------')
                        print(f'anomal_ratio : {anomal_ratio}')

                # print(f'anomal_ratio : {anomal_ratio}')

                py_serial.write("a".encode('utf-8'))
                log_json = json.dumps(log)
                db_serial.write(log_json.encode())


        display_queue.put(pose_img)
        out.write(pose_img)

def display_thread(display_queue):
    rospy.init_node('lidar_listener', anonymous = True)
    rospy.Subscriber('/scan', LaserScan, lidar_callback)
    while True:
        pose_img = display_queue.get()
        cv2.imshow('YOLO-Pose', pose_img)
        if cv2.waitKey(1) & 0xFF == 27: # if you want to quit, Press 'ESC'
            break

capture_t = threading.Thread(target=capture_thread, args=(input_queue, video_queue))
inference_t = threading.Thread(target=inference_thread, args=(input_queue, output_queue))
classification_t = threading.Thread(target=classification_thread, args=(output_queue, lidar_queue, video_queue, face_queue))

print('------------THREAD START----------')
capture_t.start()
inference_t.start()
classification_t.start()

display_thread(display_queue)
