import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import os
from PIL import Image
import pickle
import imutils.paths as paths
import face_recognition
import pickle

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import time


# from notifypy import Notify
# notification = Notify()
# notification.audio = "D:/Final_Project/facemask/yolov5-master/alarm.wav"


chtime = False
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

device = select_device('cpu')
model = attempt_load('best.pt', map_location=device)  # load FP32 model

# Get names and colors
# names = model.module.names if hasattr(model, 'module') else model.names
# print('names',names)
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()  # to FP16

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(416, s=stride)  # check img_size

#cap = cv2.VideoCapture('rtsp://192.168.240.244:8080/h264_pcm.sdp')
cap = cv2.VideoCapture(0)

data = pickle.loads(open("data/encoding1.pickle", "rb").read())

while True:
    _,img0 = cap.read()
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t1 = time_synchronized()
    pred = model(img)[0]

    pred = non_max_suppression(pred, 0.7, 0.2)
    t2 = time_synchronized()
    position = []
    for i, det in enumerate(pred):  # detections per image
        s = ''
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *xyxy, conf, cls in reversed(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                #print('cls',cls)
                # label = f'{names[int(cls)]} {conf:.2f}'
                # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                # label=np.argmax(result,axis=1)
                boxes = [(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))]
                position.append(boxes)
                print(position)
                encodings = face_recognition.face_encodings(img0, boxes)
                names = []

                for encoding in encodings:
                    matches = face_recognition.compare_faces(np.array(encoding), np.array(data["encodings"]))
                    name = "Unknown"

                    if True in matches:
                        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                        counts = {}


                        for i in matchedIdxs:
                            name = data["names"][i]
                            counts[name] = counts.get(name, 0) + 1

                            name = max(counts, key=counts.get)
                    names.append(name)




                for ((top, right, bottom, left), name) in zip(boxes, names):

                #     top = int(top * )
                #     right = int(right * 4)
                #     bottom = int(bottom * 4)
                #     left = int(left * 4)
                #     print(top)
                #     cv2.rectangle(img0, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(img0, (top, right), (bottom, left),(0, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(img0, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", img0)
    if cv2.waitKey(1) == 27:
        break



