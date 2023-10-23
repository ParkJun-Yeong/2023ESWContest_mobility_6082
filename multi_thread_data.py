# General
from datetime import datetime
from queue import Queue, Full
import threading
import cv2
import joblib
import numpy as np
# Pose
import torch
from ultralytics import YOLO
# Lidar
import rospy
from sensor_msgs.msg import LaserScan

import pandas as pd

input_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
video_queue = Queue(maxsize=1)
lidar_queue = Queue(maxsize=2)
display_queue = Queue(maxsize=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'----------{device}----------')


# rplidar a1 : sampling from -pi to pi with 1147 points
theta = np.linspace(-np.pi, np.pi, 1147) 

# Lidar
def lidar_callback(data):

    ldr = data.ranges
#    print(ldr)

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
                    torch.dist(kp[0][11], kp[0][12]).item(),
                    ])
    return skeletons

def capture_thread(input_queue, video_queue):
    cap = cv2.VideoCapture(0)
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
        results = model.predict(image, conf=0.65)
        results[0] = results[0].to(device)
        output_queue.put(results)

def classification_thread(output_queue, lidar_queue, video_queue):
    # initalize classifier
    model = joblib.load('knn_model.pkl')
    seq = []
    seq_num = 60 # prediction with 10 sequence
    anomal_ratio = 0 

    # Video writer setup
    file_path = f'results/record_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    size = (1920, 1080)
    out = cv2.VideoWriter(file_path, fourcc, 20.0, size)  # 파일명, 코덱, 프레임 수, 해상도
    while True:
        results = output_queue.get()
        pose_img = results[0].plot()

        results_lidar = lidar_queue.get()
        print('results_lidar',results_lidar)
        skeletons = extract_skeletons(results)  # Length between Landmarks
        
        # Optional: Run pose coordinates through RandomForest Classifier
        seq.append(skeletons/results_lidar)
        label = 3
        if len(seq) == 1000:
            seq = np.stack(seq)
            save_data = pd.DataFrame(seq)
            save_data['label'] = np.full(len(save_data), label)
            
            save_data.to_csv(f'data1019/{label}_01.csv')
            seq = [] 
            break
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
classification_t = threading.Thread(target=classification_thread, args=(output_queue, lidar_queue, video_queue))

capture_t.start()
inference_t.start()
classification_t.start()

display_thread(display_queue)
