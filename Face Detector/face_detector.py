# -*- coding: utf-8 -*-

#Import packages
import numpy as np
import argparse
import cv2

from imutils.video import VideoStream, FileVideoStream
import imutils
import time


#Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input image")
ap.add_argument("-p", "--prototxt", help ="path to Caffe 'deploy' prototxt file", default='deploy.prototxt.txt')
ap.add_argument("-m", "--model",  help="path to Caffe pretrained model" , default='res10_300x300_ssd_iter_140000.caffemodel')
ap.add_argument("-v", "--video",  help="path to video")
ap.add_argument("-w", "--webcam",  help="use webcam")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimun probability to filter weak detections")

args = vars(ap.parse_args())

def resize_image(image, W=500):
    (h, w) = image.shape[:2]
    scale = W/w
    newW, newH = int(w*scale), int(h*scale)
    return cv2.resize(image, (newW, newH)), newW, newH 


def detect_faces(image):
    resized, w, h = resize_image(image)
    blob = cv2.dnn.blobFromImage(resized, 1.0, (w,h), (104.0, 177.0, 123.0))
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    net.setInput(blob)
    detections = net.forward()
    return detections

def print_detections(image, detections):
    (_h, _w) = image.shape[:2]
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([_w, _h, _w, _h])
            (startX, startY, endX, endY) = box.astype("int")
            text ="{:.2f}%".format(confidence*100)
            y = startY-10 if startY-10 > 10 else startY+10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)
    
        cv2.imshow("Output", image)

def detect_image():
    image = cv2.imread(args["image"])
    print_detections(image, detect_faces(image))
    cv2.waitKey(10000)
    
def detect_webcam():
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        print_detections(frame, detect_faces(frame))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break 
    
def detect_video():
    vs = FileVideoStream(args["video"]).start()
    time.sleep(2.0)
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        print_detections(frame, detect_faces(frame))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break 
    
if args["image"]:
    detect_image()

elif args["video"]:
    detect_video()
else:
    detect_webcam()

#load image 
#image = cv2.imread(args["image"])
#print_detections(image, detect_faces(image))
    

