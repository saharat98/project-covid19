# import the necessary packages
# from imutils import face_utils
import numpy as np
import argparse
#import imutils
import dlib
import cv2
from PIL import Image
import pytesseract
import argparse
import os
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to landmark predictor file.dat")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detector", required=True,
    help="path to detector file.svm")
args = vars(ap.parse_args())
print(args)
print(args["detector"])

im_ref = np.zeros((454, 722, 3), np.uint8)
pts_ref = np.array([[77, 156], [177, 156], [77, 245], [177, 245], [54, 30], [54, 92],
    [419, 20], [609, 20], [543, 220], [705, 220], [543, 412], [705, 412],
    [8, 8], [714, 8], [714, 445], [8, 445], [84, 400], [416, 400]])


def dlibShape2numpyArray(shape):
    vec = np.empty([18, 2], dtype = int)
    for b in range(18):
        vec[b][0] = shape.part(b).x
        vec[b][1] = shape.part(b).y
    return vec

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# print(args["detector"])
detector = dlib.fhog_object_detector(args["detector"])
predictor = dlib.shape_predictor(args["shape_predictor"])
img = dlib.load_rgb_image(args["image"])
dets = detector(img)

if len(dets) > 0:
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        
        shape = predictor(img, d)
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                          shape.part(1)))
                                                  
        h, status = cv2.findHomography(dlibShape2numpyArray(shape), pts_ref, cv2.RANSAC, 5.0)
        im_out = cv2.warpPerspective(img, h, (im_ref.shape[1],im_ref.shape[0]))

        # get ID region 
        im_id = im_out[40:85, 320:590]
        
        # OCR
        id_gray = cv2.cvtColor(im_id, cv2.COLOR_BGR2GRAY)
        #id_gray = cv2.threshold(id_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #id_gray = cv2.medianBlur(id_gray, 3)
        id_gray = cv2.adaptiveThreshold(id_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,10)
        cv2.rectangle(id_gray, (1,1), (id_gray.shape[1]-1,id_gray.shape[0]-1), 255, 3)
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, id_gray)
        
        custom_config = r'--oem 1 --psm 7 digits'
        id_text = pytesseract.image_to_string(Image.open(filename), config=custom_config)
        os.remove(filename)
        print(id_text)

        im_bgr = cv2.cvtColor(im_out, cv2.COLOR_RGB2BGR)
        cv2.imwrite('temp_out.jpg', im_bgr)

        
        # Draw the face landmarks on the screen.
        win = dlib.image_window()
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        win.add_overlay(shape)
        
        win_out = dlib.image_window()
        win_out.clear_overlay()
        win_out.set_image(im_out)
        #win_out.add_overlay(dets)
        #win_out.add_overlay(shape) 
        
        win_id = dlib.image_window()
        win_id.clear_overlay()
        win_id.set_image(id_gray)        
        dlib.hit_enter_to_continue()
        

