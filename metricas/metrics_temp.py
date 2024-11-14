import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random

import os
import pandas as pd


# cargar modelo yolo from ultralytics

from ultralytics import YOLO

# cargar modelo yolo from ultralytics
model = YOLO("/home/nisla/wilfire2024/best_models/fresh-water-3/weights/best.pt")
path_videos = r"/data/nisla/data/train/"

def calculate_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area_box1 + area_box2 - intersection
    return intersection / union if union > 0 else 0


data = []

def load_label(path_frame):
    if os.path.exists(path_frame):
            with open(path_frame, "r") as f:
                line = f.readline().split()
                if len(line) == 5:
                    _, x_true, y_true, w_true, h_true = map(float, line)
                else:
                    x_true, y_true, w_true, h_true = 0, 0, 0, 0
    else:
        x_true, y_true, w_true, h_true = 0, 0, 0, 0
    return x_true, y_true, w_true, h_true


for video in os.listdir(path_videos):
    t_0, t_1, t_2, t_3, t_4 = None, None, None, None, None
    frames = os.listdir(os.path.join(path_videos, video))
    # Remove labels from the frames list
    frames = [frame for frame in frames if "txt" not in frame]
    # sort 
    # frames.sort(key=extract_frame_number)
    frames.sort()
    
    for k, frame in enumerate(frames):

        path_frame = os.path.join(path_videos, video, frame)
        path_frame = path_frame.replace(".jpg", ".txt")
        labels = []
        # cargar los 4 frames anteriores
        for i in range(0, 5):
            if k - i >= 0:
                path_frame = os.path.join(path_videos, video, frames[k - i])
                path_frame = path_frame.replace(".jpg", ".txt")
                x_true, y_true, w_true, h_true = load_label(path_frame)
                labels.append([x_true, y_true, w_true, h_true])
            else:
                labels.append([0, 0, 0, 0])

        
        
        results = model(os.path.join(path_videos, video, frame), conf=0.01)
        t_0_abs = False

        if not results or all(len(result.boxes) == 0 for result in results):
            # No detections case
            if x_true == 0:
                # no hay incendio
                label = True
            else:
                # si hay incendio
                label = False

            data.append({"video": video, "frame": frame, "confidence": 0, "box_id": 0, "iou": 0, "t_0": 0, "t_1": 0, "t_2": 0, "t_3": 0, "t_4": 0,"box": None, "gt_0": labels[0], "gt_1": labels[1], "gt_2": labels[2], "gt_3": labels[3], "gt_4": labels[4]})  
        else:
            for result in results:
                for i, box in enumerate(result.boxes):
                    bounding_box = box.xyxyn[0].tolist()
                    confidence = box.conf.tolist()[0]
                    ts = []
                    for label in labels:
                        x_true, y_true, w_true, h_true = label
                        iou = calculate_iou([x_true, y_true, x_true + w_true, y_true + h_true], bounding_box)
                        t_i = 1 if iou > 0.1 else 0
                        ts.append(t_i)

                    data.append({"video": video, "frame": frame, "confidence": confidence, "box_id": i+1,  "iou": iou, "t_0": ts[0], "t_1": ts[1], "t_2": ts[2], "t_3": ts[3], "t_4": ts[4], "box": bounding_box, "gt_0": labels[0], "gt_1": labels[1], "gt_2": labels[2], "gt_3": labels[3], "gt_4": labels[4]})

data = pd.DataFrame(data)


# export to excel
data.to_excel("data_temp.xlsx")
