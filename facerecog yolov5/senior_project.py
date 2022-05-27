# import system module
import sys

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from mainwindow import *

import numpy as np
import cv2
import pickle
import mysql.connector as con
from mysql.connector import errorcode
import datetime
import os
import time
import dlib
import os.path
import pygame
from PIL import ImageTk, Image
import pytesseract
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os
import pickle
import imutils.paths as paths
import face_recognition
import pickle
from collections import Counter

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import time

position = []
recog_count = []
check_face = 0
pygame.mixer.init()
shape_predictor = ".\\sp7.dat"
detector = ".\\ID02_1.svm"
im_ref = np.zeros((454, 722, 3), np.uint8)
pts_ref = np.array([[77, 156], [177, 156], [77, 245], [177, 245], [54, 30], [54, 92],
                    [419, 20], [609, 20], [543, 220], [705, 220], [543, 412], [705, 412],
                    [8, 8], [714, 8], [714, 445], [8, 445], [84, 400], [416, 400]])

def dlibShape2numpyArray(shape):
    vec = np.empty([18, 2], dtype=int)
    for b in range(18):
        vec[b][0] = shape.part(b).x
        vec[b][1] = shape.part(b).y
    return vec

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

data = pickle.loads(open("data/encoding1.pickle", "rb").read())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# print(args["detector"])
detector = dlib.fhog_object_detector(detector)
predictor = dlib.shape_predictor(shape_predictor)

"เชื่อมต่อฐานข้อมูล"
def connectDB():
    try:
        hosts = "127.0.0.1"
        username = "root"
        passwords = ""
        database_name = "senior_project"
        ports = 3306
        connect = con.connect(user=username, password=passwords, host=hosts, database=database_name, port=ports)
        return connect

    except con.Error as err:
        if err.errno == errorcode.ER.ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)

"แสดงข้อมูลทั้งหมดในฐานข้อมูล"
def queryData():
    connect = connectDB()
    cursor = connect.cursor()
    sql = "SELECT * FROM 2type_checkin"
    cursor.execute(sql)
    data = cursor.fetchall()
    print(data)

"เพิ่มข้อมูลสำหรับใบหน้า"
def insertData(idcard,name,pic):

    "เวลาปัจจุบัน"
    now = datetime.datetime.now()
    # now = now.strftime("%d-%m-%y %H:%M:%S")
    now_date = now.strftime("%d-%m-%y")
    now_time_colon = now.strftime("%H:%M:%S")
    now_time = now.strftime("%H-%M-%S")
    now_date_folder = now.strftime("%y-%m-%d")
    directory = now_date_folder

    connect = connectDB()
    cursor = connect.cursor()

    "บันทึกลงในฐานข้อมูล"
    sql = "INSERT INTO 2type_checkin (idcard, name, pic, date, time) VALUE ('%s', '%s', '%s', '%s', '%s')"% (idcard, name, now_time + '_' + '%s' % name + '.jpg', now_date, now_time_colon)
    cursor.execute(sql)
    connect.commit()
    connect.close()

    "เพิ่มลงในโฟลเดอร์"
    parent_dir = "Cap_Picture_Recog/"
    path = os.path.join(parent_dir, directory)
    try:
        os.mkdir(path)
        cv2.imwrite("Cap_Picture_Recog/" + "%s" % directory + "/" + "%s" % now_time + "_" + "%s" % name + ".jpg", pic)
    except FileExistsError:
        "โฟลเดอร์ถูกสร้างไว้แล้ว"
        # print("file already exists.")
        cv2.imwrite("Cap_Picture_Recog/" + "%s" % directory + "/" + "%s" % now_time + "_" + "%s" % name + ".jpg", pic)

"เพิ่มข้อมูลสำหรับบัตรประชาชน"
def insertData_idcard(idcard,name,pic):

    "เวลาปัจจุบัน"
    now = datetime.datetime.now()
    # now = now.strftime("%d-%m-%y %H:%M:%S")
    now_date = now.strftime("%d-%m-%y")
    now_time_colon = now.strftime("%H:%M:%S")
    now_time = now.strftime("%H-%M-%S")
    now_date_folder = now.strftime("%y-%m-%d")
    directory = now_date_folder

    connect = connectDB()
    cursor = connect.cursor()

    "บันทึกลงในฐานข้อมูล"
    sql = "INSERT INTO 2type_checkin (idcard, name, pic, date, time) VALUE ('%s', '%s', '%s', '%s', '%s')"% (idcard, name, pic, now_date, now_time_colon)
    cursor.execute(sql)
    connect.commit()
    connect.close()

"ลบข้อมูล"
def deleteData(idcard):
    connect = connectDB()
    cursor = connect.cursor()
    sql = "DELETE FROM 2type_checkin WHERE idcard='%d'"% idcard
    cursor.execute(sql)
    connect.commit()
    connect.close()

"หา idcard ว่าคนนี้เข้ามาเวลาไหนบ้าง"
def idcard_search(idcard):
    connect = connectDB()
    cursor = connect.cursor()
    sql = "SELECT * FROM 2type_checkin WHERE idcard='%d'"% idcard
    cursor.execute(sql)
    result = cursor.fetchall()

    for x in result:
        print(x)

"หาวันที่ว่าวันนี้มีใครเข้ามาบ้าง"
def date_search(date):
    connect = connectDB()
    cursor = connect.cursor()
    sql = "SELECT * FROM 2type_checkin WHERE date='%s'" % date #DD-MM-YY
    cursor.execute(sql)
    result = cursor.fetchall()

    for x in result:
        print(x)

"หาชื่อที่มากสุดในลิสท์"
def most_frequent(List):
    counter = 0
    the_most = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            the_most = i

    return the_most

class MainWindow(QWidget):

    # class constructor
    "import ui"
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        # self.timer.timeout.connect(self.Idcardread)
        # set control_bt callback clicked  function
        self.ui.idcardread.toggled.connect(self.Controltimer_to_idcardread)
        # self.ui.on_idcardread.clicked.connect(self.Controltimer_to_idcardread)
        self.ui.facerec.toggled.connect(self.Controltimer_to_facerec)
        # self.ui.on_facerec.clicked.connect(self.Controltimer_to_facerec)
        self.ui.idcardandfacerec.toggled.connect(self.Controltimer_to_idcardandfacerec)

    "เปิดกล้อง webcam"
    # view camera
    def Idcardread(self):

        "เวลาปัจจุบัน"
        now = datetime.datetime.now()
        # now = now.strftime("%d-%m-%y %H:%M:%S")
        now_date = now.strftime("%d-%m-%y")
        now_time_colon = now.strftime("%H:%M:%S")
        now_time = now.strftime("%H-%M-%S")
        now_date_folder = now.strftime("%y-%m-%d")
        directory = now_date_folder

        self.ui.checkin_status.setText("กรุณาแสดงบัตรประชาชน")

        TIMER = int(3)
        # read image in BGR format
        ret, frame = self.cap.read()
        cv2.imwrite("Cap_Picture/Cap_Pic" + ".jpg", frame)
        # convert image to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get image infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.label.setPixmap(QPixmap.fromImage(qImg))

        image = "Cap_Picture\\Cap_Pic.jpg"
        img = dlib.load_rgb_image(image)
        dets = detector(img)

        if len(dets) > 0:

            pygame.mixer.music.load("sound/idcardfor3sec.mp3")
            self.ui.checkin_status.setText("กรุณาแสดงบัตรค้างไว้ 3 วินาที")
            pygame.mixer.music.play()
            prev = time.time()

            while TIMER >= 0:

                ret, frame = self.cap.read()
                # ret, frame = cap.read()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Display countdown on each frame
                # specify the font and draw the
                # countdown using puttext
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(TIMER),
                    (200, 250), font,
                    7, (0, 255, 255),
                    4, cv2.LINE_AA)
                # cv2.imshow('frame', frame)
                # get image infos
                height, width, channel = frame.shape
                step = channel * width
                # create QImage from image
                qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                # show image in img_label
                self.ui.label.setPixmap(QPixmap.fromImage(qImg))
                cv2.waitKey(1)

                # current time
                cur = time.time()

                # Update and keep track of Countdown
                # if time elapsed is one second
                # than decrese the counter
                if cur - prev >= 1:
                    prev = cur
                    TIMER = TIMER - 1

            else:
                # ret, frame = cap.read()
                ret, frame = self.cap.read()

                # Display the clicked frame for 2
                # sec.You can increase time in
                # waitKey also
                # cv2.imshow('Delay_Pic', frame)

                # time for which image displayed


                # Save the frame
                cv2.imwrite('Cap_Picture/Delay_Pic.jpg', frame)
                cv2.waitKey(1000)
                image = "Cap_Picture\\Delay_Pic.jpg"
                img = dlib.load_rgb_image(image)
                dets = detector(img)

                if len(dets) > 0:

                    for k, d in enumerate(dets):
                        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                        #    k, d.left(), d.top(), d.right(), d.bottom()))
                        # Get the landmarks/parts for the face in box d.

                        shape = predictor(img, d)
                        # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                        #                                          shape.part(1)))

                        h, status = cv2.findHomography(dlibShape2numpyArray(shape), pts_ref, cv2.RANSAC, 5.0)
                        im_out = cv2.warpPerspective(img, h, (im_ref.shape[1], im_ref.shape[0]))

                        # get ID region
                        im_id = im_out[40:85, 320:590]

                        # OCR
                        id_gray = cv2.cvtColor(im_id, cv2.COLOR_BGR2GRAY)
                        # id_gray = cv2.threshold(id_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        # id_gray = cv2.medianBlur(id_gray, 3)
                        id_gray = cv2.adaptiveThreshold(id_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25,10)
                        cv2.rectangle(id_gray, (1, 1), (id_gray.shape[1] - 1, id_gray.shape[0] - 1), 255, 3)
                        # filename = "{}.png".format(os.getpid())
                        # cv2.imwrite(filename, id_gray)

                        custom_config = r'--oem 1 --psm 7 digits'
                        id_text = pytesseract.image_to_string(id_gray, config=custom_config)
                        # print(len(str(int(id_text))))

                        if len(str(id_text)) == 15:
                            # if len(str(id_text)) != 13:
                            # os.remove(filename)
                            id_text = int(id_text)

                            print("11")
                            im_bgr = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                            print("12")
                            cv2.imwrite('Cap_Picture/IDCard_Detect.jpg', im_bgr)
                            cv2.waitKey(2000)

                            if os.path.exists("Cap_Picture/IDCard_Detect.jpg"):
                                if len(str(id_text)) == 13:
                                    # print(id_text)
                                    insertData_idcard(id_text, "-", "-")
                                    self.ui.idcard_checkin.setText(str(id_text))
                                    self.ui.name_checkin.setText("-")
                                    self.ui.time_checkin.setText(now_time_colon)
                                    self.ui.date_checkin.setText(now_date)
                                    self.ui.photo.setPixmap(QtGui.QPixmap("Cap_Picture/IDCard_Detect.jpg"))
                                    pygame.mixer.music.load("sound/savesuccess.mp3")
                                    pygame.mixer.music.play()
                                    print("บันทึกข้อมูลบัตรบัตรเรียบร้อยแล้ว")
                                    self.ui.checkin_status.setText("บันทึกข้อมูลเรียบร้อยแล้ว")
                                    os.remove("Cap_Picture/IDCard_Detect.jpg")
                                    # os.remove("Cap_Picture/Delay_Pic.jpg")
                                else:
                                    print("กรุณาลองอีกครั้ง")
                                    pygame.mixer.music.load("sound/tryagain.mp3")
                                    pygame.mixer.music.play()
                                    print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                                    self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")
                                    os.remove("Cap_Picture/IDCard_Detect.jpg")
                            else:
                                print("กรุณาลองอีกครั้ง")
                                pygame.mixer.music.load("sound/tryagain.mp3")
                                pygame.mixer.music.play()
                                print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                                self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")
                        # os.remove("Cap_Picture/Delay_Pic.jpg")
                        # cv2.waitKey(1000)
                        # os.remove("Cap_Picture/Delay_Pic.jpg")
                        # cv2.waitKey(1000)
                        # cv2.destroyAllWindows()
                        else:
                            print("กรุณาลองอีกครั้ง")
                            pygame.mixer.music.load("sound/tryagain.mp3")
                            pygame.mixer.music.play()
                            print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                            self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")
                else:
                    print("กรุณาลองอีกครั้ง")
                    pygame.mixer.music.load("sound/tryagain.mp3")
                    pygame.mixer.music.play()
                    print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                    self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")

                # cv2.waitKey(1000)
                # if os.path.exists("Cap_Picture/IDCard_Detect.jpg"):
                #     self.ui.photo.setPixmap(QtGui.QPixmap("Cap_Picture/IDCard_Detect.jpg"))
                #     pygame.mixer.music.load("sound/savesuccess.mp3")
                #     pygame.mixer.music.play()
                #     print("บันทึกข้อมูลบัตรบัตรเรียบร้อยแล้ว")
                #     self.ui.checkin_status.setText("บันทึกข้อมูลเรียบร้อยแล้ว")
                #     os.remove("Cap_Picture/IDCard_Detect.jpg")
                # else:
                #     pygame.mixer.music.load("sound/tryagain.mp3")
                #     pygame.mixer.music.play()
                #     print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                #     self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")

                # HERE we can reset the Countdown timer
                # if we want more Capture without closing
                # the camera

            cv2.waitKey(2000)
            os.remove("Cap_Picture/Delay_Pic.jpg")
            # self.ui.photo.setPixmap(QtGui.QPixmap("pic/idcard_pic1.jpg"))
            self.ui.idcard_checkin.setText("-")
            self.ui.name_checkin.setText("-")
            self.ui.time_checkin.setText("-")
            self.ui.date_checkin.setText("-")

        # get image infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.label.setPixmap(QPixmap.fromImage(qImg))

    def Facerec(self):

        global position
        global check_face
        global recog_count

        "เวลาปัจจุบัน"
        now = datetime.datetime.now()
        # now = now.strftime("%d-%m-%y %H:%M:%S")
        now_date = now.strftime("%d-%m-%y")
        now_time_colon = now.strftime("%H:%M:%S")
        now_time = now.strftime("%H-%M-%S")
        now_date_folder = now.strftime("%y-%m-%d")
        directory = now_date_folder

        self.ui.checkin_status.setText("กรุณาแสดงใบหน้า")

        TIMER = int(3)
        # num = int
        # read image in BGR format
        ret, frame = self.cap.read()
        img = letterbox(frame, imgsz, stride=stride)[0]
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
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    # print('cls',cls)
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
                    # label=np.argmax(result,axis=1)
                    boxes = [(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))]
                    position.append(boxes)
                    # print(position)
                    encodings = face_recognition.face_encodings(frame, boxes)
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
                                print(counts)

                                name = max(counts, key=counts.get)
                        names.append(name)
                        recog_count.append(name)
                        position.append(name)
                        check_face = check_face + 1
                        # print(check_face)
                        # print(name)

                    for ((top, right, bottom, left), name) in zip(boxes, names):
                        #     top = int(top * )
                        #     right = int(right * 4)
                        #     bottom = int(bottom * 4)
                        #     left = int(left * 4)
                        #     print(top)
                        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(frame, (top, right), (bottom, left), (0, 255, 0), 2)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    if check_face == 3:

                        if most_frequent(recog_count) != "Unknown":

                            pygame.mixer.music.load("sound/staystill3sec.mp3")
                            self.ui.checkin_status.setText("กรุณาแสดงใบหน้าค้างไว้ 3 วินาที")
                            pygame.mixer.music.play()
                            prev = time.time()

                            while TIMER >= 0:

                                ret, frame = self.cap.read()
                                # ret, frame = cap.read()

                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                # Display countdown on each frame
                                # specify the font and draw the
                                # countdown using puttext
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(frame, str(TIMER),
                                            (200, 250), font,
                                            7, (0, 255, 255),
                                            4, cv2.LINE_AA)
                                # cv2.imshow('frame', frame)
                                # get image infos
                                height, width, channel = frame.shape
                                step = channel * width
                                # create QImage from image
                                qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                                # show image in img_label
                                self.ui.label.setPixmap(QPixmap.fromImage(qImg))
                                cv2.waitKey(1)

                                # current time
                                cur = time.time()

                                # Update and keep track of Countdown
                                # if time elapsed is one second
                                # than decrese the counter
                                if cur - prev >= 1:
                                    prev = cur
                                    TIMER = TIMER - 1
                            else:
                                # ret, frame = cap.read()
                                ret, frame = self.cap.read()
                                insertData("-", name, frame)
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                height, width, channel = frame.shape
                                step = channel * width
                                # create QImage from image
                                qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                                # show image in img_label
                                # self.ui.label.setPixmap(QPixmap.fromImage(qImg))

                                # Display the clicked frame for 2
                                # sec.You can increase time in
                                # waitKey also
                                # cv2.imshow('Delay_Pic', frame)

                                # time for which image displayed

                                # Save the frame
                                pygame.mixer.music.load("sound/savesuccess.mp3")
                                pygame.mixer.music.play()
                                print("บันทึกใบหน้าเรียบร้อยแล้ว")
                                self.ui.checkin_status.setText("บันทึกข้อมูลเรียบร้อยแล้ว")
                                self.ui.face.setPixmap(QtGui.QPixmap(qImg))
                                self.ui.idcard_checkin.setText("-")
                                self.ui.name_checkin.setText(name)
                                self.ui.time_checkin.setText(now_time_colon)
                                self.ui.date_checkin.setText(now_date)

                                # HERE we can reset the Countdown timer
                                # if we want more Capture without closing
                                # the camera

                            cv2.waitKey(2000)
                            # self.ui.face.setPixmap(QtGui.QPixmap("pic/face_cartoon.jpg"))
                            self.ui.idcard_checkin.setText("-")
                            self.ui.name_checkin.setText("-")
                            self.ui.time_checkin.setText("-")
                            self.ui.date_checkin.setText("-")
                            print(most_frequent(recog_count))
                            print(recog_count)
                            recog_count = []
                            position = []
                            check_face = 0

                        elif most_frequent(recog_count) == "Unknown":
                            # elif fake == "Unknown":
                            pygame.mixer.music.load("sound/tryagain_face.mp3")
                            pygame.mixer.music.play()
                            self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จกรุณาแสดงใบหน้าอีกครั้ง")
                            print(most_frequent(recog_count))
                            print(recog_count)
                            recog_count = []
                            position = []
                            check_face = 0
                            cv2.waitKey(1000)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.label.setPixmap(QPixmap.fromImage(qImg))

    def idcardandfacerec(self):

        global position
        global check_face
        global recog_count

        "เวลาปัจจุบัน"
        now = datetime.datetime.now()
        # now = now.strftime("%d-%m-%y %H:%M:%S")
        now_date = now.strftime("%d-%m-%y")
        now_time_colon = now.strftime("%H:%M:%S")
        now_time = now.strftime("%H-%M-%S")
        now_date_folder = now.strftime("%y-%m-%d")
        directory = now_date_folder

        self.ui.checkin_status.setText("กรุณาแสดงใบหน้า")

        TIMER = int(3)
        # num = int
        # read image in BGR format
        ret, frame = self.cap.read()
        img = letterbox(frame, imgsz, stride=stride)[0]
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
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    # print('cls',cls)
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], line_thickness=3)
                    # label=np.argmax(result,axis=1)
                    boxes = [(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))]
                    position.append(boxes)
                    # print(position)
                    encodings = face_recognition.face_encodings(frame, boxes)
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
                        recog_count.append(name)
                        position.append(name)
                        check_face = check_face + 1
                        # print(check_face)
                        # print(name)

                    for ((top, right, bottom, left), name) in zip(boxes, names):
                        #     top = int(top * )
                        #     right = int(right * 4)
                        #     bottom = int(bottom * 4)
                        #     left = int(left * 4)
                        #     print(top)
                        #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(frame, (top, right), (bottom, left), (0, 255, 0), 2)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                    if check_face == 3:

                        # print(check_face)
                        # print(name)
                        # cnt = Counter(position)
                        # print(cnt.most_common(1)[0][0])
                        # print(cnt.most_common(1))
                        # print(most_frequent(recog_count))
                        # fake = "Unknown"
                        # print(fake)
                        # print(recog_count)
                        # recog_count = []
                        # position = []
                        # check_face = 0
                        fake = "Unknown"
                        # print(fake)

                        # if most_frequent(recog_count) == "Unknown":
                        if most_frequent(recog_count) != "Unknown":

                            pygame.mixer.music.load("sound/staystill3sec.mp3")
                            self.ui.checkin_status.setText("กรุณาแสดงใบหน้าค้างไว้ 3 วินาที")
                            pygame.mixer.music.play()
                            prev = time.time()

                            while TIMER >= 0:

                                ret, frame = self.cap.read()
                                # ret, frame = cap.read()

                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                # Display countdown on each frame
                                # specify the font and draw the
                                # countdown using puttext
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(frame, str(TIMER),
                                            (200, 250), font,
                                            7, (0, 255, 255),
                                            4, cv2.LINE_AA)
                                # cv2.imshow('frame', frame)
                                # get image infos
                                height, width, channel = frame.shape
                                step = channel * width
                                # create QImage from image
                                qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                                # show image in img_label
                                self.ui.label.setPixmap(QPixmap.fromImage(qImg))
                                cv2.waitKey(1)

                                # current time
                                cur = time.time()

                                # Update and keep track of Countdown
                                # if time elapsed is one second
                                # than decrese the counter
                                if cur - prev >= 1:
                                    prev = cur
                                    TIMER = TIMER - 1
                            else:
                                # ret, frame = cap.read()
                                ret, frame = self.cap.read()
                                insertData("-", name, frame)
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                height, width, channel = frame.shape
                                step = channel * width
                                # create QImage from image
                                qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                                # show image in img_label
                                # self.ui.label.setPixmap(QPixmap.fromImage(qImg))

                                # Display the clicked frame for 2
                                # sec.You can increase time in
                                # waitKey also
                                # cv2.imshow('Delay_Pic', frame)

                                # time for which image displayed

                                # Save the frame
                                pygame.mixer.music.load("sound/savesuccess.mp3")
                                pygame.mixer.music.play()
                                print("บันทึกใบหน้าเรียบร้อยแล้ว")
                                self.ui.checkin_status.setText("บันทึกข้อมูลเรียบร้อยแล้ว")
                                self.ui.face.setPixmap(QtGui.QPixmap(qImg))
                                self.ui.idcard_checkin.setText("-")
                                self.ui.name_checkin.setText(name)
                                self.ui.time_checkin.setText(now_time_colon)
                                self.ui.date_checkin.setText(now_date)

                                # HERE we can reset the Countdown timer
                                # if we want more Capture without closing
                                # the camera

                            cv2.waitKey(2000)
                            # self.ui.face.setPixmap(QtGui.QPixmap("pic/face_cartoon.jpg"))
                            self.ui.idcard_checkin.setText("-")
                            self.ui.name_checkin.setText("-")
                            self.ui.time_checkin.setText("-")
                            self.ui.date_checkin.setText("-")
                            # print(most_frequent(recog_count))
                            # print(recog_count)
                            recog_count = []
                            position = []
                            check_face = 0

                        elif most_frequent(recog_count) == "Unknown":
                        # elif fake == "Unknown":
                            pygame.mixer.music.load("sound/noinfor.mp3")
                            pygame.mixer.music.play()
                            self.ui.checkin_status.setText("ไม่พบข้อมูลกรุณาแสดงบัตรประชาชน")
                            cv2.waitKey(1000)

                            # self.timer2.start()
                            # self.cap.release()
                            # self.timer2.timeout.disconnect(self.idcardandfacerec)
                            # self.timer.start(20)
                            # self.timer.timeout.connect(self.noinfo_face)

                            # while fake == "Unknown":
                            while most_frequent(recog_count) == "Unknown":
                                "เวลาปัจจุบัน"
                                now = datetime.datetime.now()
                                # now = now.strftime("%d-%m-%y %H:%M:%S")
                                now_date = now.strftime("%d-%m-%y")
                                now_time_colon = now.strftime("%H:%M:%S")
                                now_time = now.strftime("%H-%M-%S")
                                now_date_folder = now.strftime("%y-%m-%d")
                                directory = now_date_folder

                                self.ui.checkin_status.setText("กรุณาแสดงบัตรประชาชน")

                                TIMER = int(3)
                                # read image in BGR format
                                ret, frame = self.cap.read()
                                cv2.imwrite("Cap_Picture/Cap_Pic" + ".jpg", frame)
                                # convert image to RGB format
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                                # get image infos
                                height, width, channel = frame.shape
                                step = channel * width
                                # create QImage from image
                                qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                                # show image in img_label
                                self.ui.label.setPixmap(QPixmap.fromImage(qImg))
                                cv2.waitKey(1)

                                image = "Cap_Picture\\Cap_Pic.jpg"
                                img = dlib.load_rgb_image(image)
                                dets = detector(img)

                                if len(dets) > 0:

                                    pygame.mixer.music.load("sound/idcardfor3sec.mp3")
                                    self.ui.checkin_status.setText("กรุณาแสดงบัตรค้างไว้ 3 วินาที")
                                    pygame.mixer.music.play()
                                    prev = time.time()

                                    while TIMER >= 0:

                                        ret, frame = self.cap.read()
                                        # ret, frame = cap.read()

                                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        # Display countdown on each frame
                                        # specify the font and draw the
                                        # countdown using puttext
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        cv2.putText(frame, str(TIMER),
                                                    (200, 250), font,
                                                    7, (0, 255, 255),
                                                    4, cv2.LINE_AA)
                                        # cv2.imshow('frame', frame)
                                        # get image infos
                                        height, width, channel = frame.shape
                                        step = channel * width
                                        # create QImage from image
                                        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                                        # show image in img_label
                                        self.ui.label.setPixmap(QPixmap.fromImage(qImg))
                                        cv2.waitKey(1)

                                        # current time
                                        cur = time.time()

                                        # Update and keep track of Countdown
                                        # if time elapsed is one second
                                        # than decrese the counter
                                        if cur - prev >= 1:
                                            prev = cur
                                            TIMER = TIMER - 1

                                    else:
                                        # ret, frame = cap.read()
                                        ret, frame = self.cap.read()

                                        # Display the clicked frame for 2
                                        # sec.You can increase time in
                                        # waitKey also
                                        # cv2.imshow('Delay_Pic', frame)

                                        # time for which image displayed

                                        # Save the frame
                                        cv2.imwrite('Cap_Picture/Delay_Pic.jpg', frame)
                                        cv2.waitKey(1000)
                                        image = "Cap_Picture\\Delay_Pic.jpg"
                                        img = dlib.load_rgb_image(image)
                                        dets = detector(img)

                                        if len(dets) > 0:

                                            for k, d in enumerate(dets):
                                                # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                                                #    k, d.left(), d.top(), d.right(), d.bottom()))
                                                # Get the landmarks/parts for the face in box d.

                                                shape = predictor(img, d)
                                                # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                #                                          shape.part(1)))

                                                h, status = cv2.findHomography(dlibShape2numpyArray(shape), pts_ref,cv2.RANSAC, 5.0)
                                                im_out = cv2.warpPerspective(img, h, (im_ref.shape[1], im_ref.shape[0]))

                                                # get ID region
                                                im_id = im_out[40:85, 320:590]

                                                # OCR
                                                id_gray = cv2.cvtColor(im_id, cv2.COLOR_BGR2GRAY)
                                                # id_gray = cv2.threshold(id_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                                                # id_gray = cv2.medianBlur(id_gray, 3)
                                                id_gray = cv2.adaptiveThreshold(id_gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 25, 10)
                                                cv2.rectangle(id_gray, (1, 1),(id_gray.shape[1] - 1, id_gray.shape[0] - 1), 255, 3)
                                                # filename = "{}.png".format(os.getpid())
                                                # cv2.imwrite(filename, id_gray)

                                                custom_config = r'--oem 1 --psm 7 digits'
                                                id_text = pytesseract.image_to_string(id_gray, config=custom_config)
                                                # print(len(str(int(id_text))))

                                                if len(str(id_text)) == 15:
                                                    # if len(str(id_text)) != 13:
                                                    # os.remove(filename)
                                                    id_text = int(id_text)

                                                    print("11")
                                                    im_bgr = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
                                                    print("12")
                                                    cv2.imwrite('Cap_Picture/IDCard_Detect.jpg', im_bgr)
                                                    cv2.waitKey(2000)

                                                    if os.path.exists("Cap_Picture/IDCard_Detect.jpg"):
                                                        if len(str(id_text)) == 13:
                                                            # print(id_text)
                                                            insertData_idcard(id_text, "-", "-")
                                                            self.ui.idcard_checkin.setText(str(id_text))
                                                            self.ui.name_checkin.setText("-")
                                                            self.ui.time_checkin.setText(now_time_colon)
                                                            self.ui.date_checkin.setText(now_date)
                                                            self.ui.photo.setPixmap(QtGui.QPixmap("Cap_Picture/IDCard_Detect.jpg"))
                                                            pygame.mixer.music.load("sound/savesuccess.mp3")
                                                            pygame.mixer.music.play()
                                                            print("บันทึกข้อมูลบัตรบัตรเรียบร้อยแล้ว")
                                                            self.ui.checkin_status.setText("บันทึกข้อมูลเรียบร้อยแล้ว")
                                                            os.remove("Cap_Picture/IDCard_Detect.jpg")
                                                            # os.remove("Cap_Picture/Delay_Pic.jpg")
                                                            # fake = "Do"
                                                            recog_count = []
                                                            position = []
                                                            check_face = 0
                                                        else:
                                                            print("กรุณาลองอีกครั้ง")
                                                            pygame.mixer.music.load("sound/tryagain.mp3")
                                                            pygame.mixer.music.play()
                                                            print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                                                            self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")
                                                            os.remove("Cap_Picture/IDCard_Detect.jpg")
                                                    else:
                                                        print("กรุณาลองอีกครั้ง")
                                                        pygame.mixer.music.load("sound/tryagain.mp3")
                                                        pygame.mixer.music.play()
                                                        print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                                                        self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")
                                                # os.remove("Cap_Picture/Delay_Pic.jpg")
                                                # cv2.waitKey(1000)
                                                # os.remove("Cap_Picture/Delay_Pic.jpg")
                                                # cv2.waitKey(1000)
                                                # cv2.destroyAllWindows()
                                                else:
                                                    print("กรุณาลองอีกครั้ง")
                                                    pygame.mixer.music.load("sound/tryagain.mp3")
                                                    pygame.mixer.music.play()
                                                    print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                                                    self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")
                                        else:
                                            print("กรุณาลองอีกครั้ง")
                                            pygame.mixer.music.load("sound/tryagain.mp3")
                                            pygame.mixer.music.play()
                                            print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                                            self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")

                                        # cv2.waitKey(1000)
                                        # if os.path.exists("Cap_Picture/IDCard_Detect.jpg"):
                                        #     self.ui.photo.setPixmap(QtGui.QPixmap("Cap_Picture/IDCard_Detect.jpg"))
                                        #     pygame.mixer.music.load("sound/savesuccess.mp3")
                                        #     pygame.mixer.music.play()
                                        #     print("บันทึกข้อมูลบัตรบัตรเรียบร้อยแล้ว")
                                        #     self.ui.checkin_status.setText("บันทึกข้อมูลเรียบร้อยแล้ว")
                                        #     os.remove("Cap_Picture/IDCard_Detect.jpg")
                                        # else:
                                        #     pygame.mixer.music.load("sound/tryagain.mp3")
                                        #     pygame.mixer.music.play()
                                        #     print("บันทึกข้อมูลบัตรไม่สำเร็จ")
                                        #     self.ui.checkin_status.setText("บันทึกข้อมูลไม่สำเร็จ กรุณาแสดงบัตรอีกครั้ง")

                                        # HERE we can reset the Countdown timer
                                        # if we want more Capture without closing
                                        # the camera

                                    cv2.waitKey(2000)
                                    os.remove("Cap_Picture/Delay_Pic.jpg")
                                    # self.ui.photo.setPixmap(QtGui.QPixmap("pic/idcard_pic1.jpg"))
                                    self.ui.idcard_checkin.setText("-")
                                    self.ui.name_checkin.setText("-")
                                    self.ui.time_checkin.setText("-")
                                    self.ui.date_checkin.setText("-")

                                # # get image infos
                                # height, width, channel = frame.shape
                                # step = channel * width
                                # # create QImage from image
                                # qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
                                # # show image in img_label
                                # self.ui.label.setPixmap(QPixmap.fromImage(qImg))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        self.ui.label.setPixmap(QPixmap.fromImage(qImg))

    "จับเวลาเปิดปิดกล้อง"
    # start/stop timer
    def Controltimer_to_idcardread(self):

        if self.ui.idcardread.isChecked():
            # if timer is stopped
            if not self.timer.isActive():
                self.ui.status.setText("สแกนเฉพาะบัตร")
                # create video capture
                self.cap = cv2.VideoCapture(0)
                # start timer
                self.timer.start(20)
                self.timer.timeout.connect(self.Idcardread)
        else:
            self.timer.stop()
            self.cap.release()
            self.timer.timeout.disconnect(self.Idcardread)

    def Controltimer_to_facerec(self):

        global position
        global recog_count
        global check_face

        position = []
        recog_count = []
        check_face = 0

        if self.ui.facerec.isChecked():
            # if timer is stopped
            if not self.timer.isActive():
                self.ui.status.setText("สแกนเฉพาะใบหน้า")
                # create video capture
                self.cap = cv2.VideoCapture(0)
                # start timer
                self.timer.start(20)
                self.timer.timeout.connect(self.Facerec)

        else:
            self.timer.stop()
            self.cap.release()
            self.timer.timeout.disconnect(self.Facerec)

    def Controltimer_to_idcardandfacerec(self):

        global position
        global recog_count
        global check_face

        position = []
        recog_count = []
        check_face = 0

        if self.ui.idcardandfacerec.isChecked():
            # if timer is stopped
            if not self.timer.isActive():
                self.ui.status.setText("สแกนบัตรและใบหน้า")
                # create video capture
                self.cap = cv2.VideoCapture(0)
                # start timer
                self.timer.start(20)
                self.timer.timeout.connect(self.idcardandfacerec)
        else:
            self.timer.stop()
            self.cap.release()
            self.timer.timeout.disconnect(self.idcardandfacerec)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()


    sys.exit(app.exec_())