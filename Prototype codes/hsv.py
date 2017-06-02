#!/usr/bin/env python
# -*- coding: utf-8 -*-
# returns HSV value of the pixel under the cursor in a video stream
# author: achuwilson
# achuwilson.wordpress.com
import cv
import cv2
import time
import numpy
x_co = 0
y_co = 0
hmin=[100,255,255]
hmax=[0,0,0]
def on_mouse(event,x,y,flag,param):
  global x_co
  global y_co
  if(event==cv.CV_EVENT_MOUSEMOVE):
    x_co=x
    y_co=y

cv.NamedWindow("camera", 1)
font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 1, 0, 2, 8)
capture1 = cv2.VideoCapture(0)
ret,camera = capture1.read()
cv2.imshow("camera", camera)
cv2.waitKey(10000)
capture1.release()
capture = cv.CaptureFromCAM(0)
while True:
    src = cv.QueryFrame(capture)
    cv.Smooth(src, src, cv.CV_BLUR, 3)
    hsv = cv.CreateImage(cv.GetSize(src), 8, 3)
    thr = cv.CreateImage(cv.GetSize(src), 8, 1)
    cv.CvtColor(src, hsv, cv.CV_BGR2HSV)
    cv.SetMouseCallback("camera",on_mouse, 0);
    s=cv.Get2D(hsv,y_co,x_co)
    if hmin[0]>s[0]:
      hmin[0]=s[0]
    if hmin[1]>s[1]:
      hmin[1]=s[1]
    if hmin[2]>s[2]:
      hmin[2]=s[2]
    if hmax[0]<s[0]:
      hmax[0]=s[0]
    if hmax[1]<s[1]:
      hmax[1]=s[1]
    if hmax[2]<s[2]:
      hmax[2]=s[2]
    print "min:",hmin[0], "  ",hmin[1],"  ",hmin[2],"          max:",hmax[0], "  ",hmax[1],"  ",hmax[2]
    cv.PutText(src,str(s[0])+","+str(s[1])+","+str(s[2]), (x_co,y_co),font, (55,25,255))
    cv.ShowImage("camera", src)
    if cv.WaitKey(10) == 27:
        break
