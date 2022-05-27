import argparse
import time
from pathlib import Path

import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

device = select_device('')
model = attempt_load('best.pt', map_location=device)  # load FP32 model

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
print('names',names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
        model.half()  # to FP16

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(416, s=stride)  # check img_size
path = 'D:/Final_Project/facemask/yolov3/test2/without_mask'
os.chdir(path)
imgList = os.listdir()
# print(imgList)
acc_count = 0
for i,im in enumerate(imgList):
    # if (79<i<90):
        
    dataset = LoadImages(im, img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # t1 = time_synchronized()
        pred = model(img)[0]
        # print(pred)
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45)
        # t2 = time_synchronized()
        # print(pred)
        # Process detections
        for j, det in enumerate(pred):  # detections per image
            # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            #p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            # print()
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Rescale boxes from img_size to im0 size
                    # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                    # print('cls',cls)
                    if im[:4] == names[int(cls)]:
                        acc_count+=1
                    label = f'{im} --> {names[int(cls)]} {conf:.2f}'
                    # text.append(label)
                    print(label)
                    # print('{} --> {}'.format(im,label))
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # cv2.imshow(str(p), im0)
                # cv2.waitKey(0)
# for i in text:
#     print(i)
print('Accuracy: {}'.format(acc_count/len(imgList)))