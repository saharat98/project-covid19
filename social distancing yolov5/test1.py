import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import math

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import time

from pygame import mixer
mixer.init()
sound = mixer.Sound('dis.wav')
chtime = False


w_real = 420
h_real = 360
tl = (254,294)
tr = (834,303)
br = (1094,577)
bl = (38,588)

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


def dis_tance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def calc_real_pos(w_real,h_real,w_px,h_px,x,y):
    return w_real*x/w_px,h_real*y/h_px

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def click_windows(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse down X: {} Y: {}".format(x,y))
        
def four_point_transform(image):
	# obtain a consistent order of the points and unpack them
	# individually
	#rect = order_points(pts)
	#(tl, tr, br, bl) = rect
	
	rect = np.float32([tl,tr,br,bl])
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped,M

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
#parser.add_argument('--source', type=str, default='progess.MOV', help='source')
opt = parser.parse_args()

set_logging()
device = select_device()
model = attempt_load('yolov5s.pt', map_location=device)  # load FP32 model ++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
print('device : ',device.type)
print('names',names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

half = device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()  # to FP16

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(448, s=stride)  # check img_size


cap = cv2.VideoCapture(0)#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# cap = cv2.VideoCapture(0)
color = [random.randint(0, 255) for _ in range(3)]
while True:
    cap_check,img0 = cap.read()
    if not cap_check:
        break
    #img0 = cv2.imread("C:/Users/wiraw/Desktop/distance/image/1.jpg")
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

    #pred = non_max_suppression(pred, 0.7, 0.7)
    pred = non_max_suppression(pred,0.7)
    t2 = time_synchronized()
    #print('')
    
    l_person = []
    for i, det in enumerate(pred):  # detections per image
        s = ''
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
                if f'{names[int(cls)]}' == 'person':
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    #print(label,c1, c2)
                    l_person.append([label,c1, c2])
    #print(f'{s}Done. ({t2 - t1:.3f}s)')
    color = color or [random.randint(0, 255) for _ in range(3)]
    tl_ = round(0.002 * (img0.shape[0] + img0.shape[1]) / 2) + 1  # line/font thickness

    listp = []
    for i in range(len(l_person)):
        #l_person[i][0] : label
        #l_person[i][1] : point1 x,y
        #l_person[i][2] : point2 x+w,y+h
        cv2.rectangle(img0,l_person[i][1],l_person[i][2],color,thickness=tl_, lineType=cv2.LINE_AA)
        w__ = l_person[i][2][0] - l_person[i][1][0]
        x__ = l_person[i][1][0]
        #print(w__,x__,l_person[i][2][1])
        listp.append([ int( (w__/2) + x__ ),int(l_person[i][2][1])])
    checkH = []
    img_wrap,H = four_point_transform(img0)
    img_black = np.zeros((img_wrap.shape[0],img_wrap.shape[1]),np.float32)
    # print(H)
    #print(listp)



    a = np.array(listp,dtype='float32')


    if len(listp) == 0:
        listout = []
    else:
        listout = cv2.perspectiveTransform(a[None,:,:],H)
    #print(listout)

        w_px = img_wrap.shape[1]
        h_px = img_wrap.shape[0]

        #print(w,h)
        list_pp_pos = []
        for x in listout[0]:
            #print(x)
            cv2.circle(img_black,(int(x[0]),int(x[1])),3,(255,255,255),-1)
            # cv2.imshow('result1', img_black)
            list_pp_pos.append(calc_real_pos(w_real,h_real,w_px,h_px,int(x[0]),int(x[1])))

        noti = False
        if len(list_pp_pos)>1:
            for x1 in range(len(list_pp_pos)):
                for x2 in range(x1+1,len(list_pp_pos)):
                    # print(x1,x2)
                    d = dis_tance(list_pp_pos[x1][0],list_pp_pos[x1][1],list_pp_pos[x2][0],list_pp_pos[x2][1])
                    # print('distance',d)
                    if d<200:
                        noti = True

        if noti:
            if  chtime == False:
                chtime = True
                delay = time.time()
                sound.play()
                print('Beep')
            if time.time() - delay > 3 :
                chtime = False

    cv2.imshow('result', img0)
    # cv2.imshow('img_black', img_black)
    # cv2.imshow('img_wrap', cv2.resize( img_wrap,(w_real,h_real)))
    if cv2.waitKey(1) == 27: 
        break
