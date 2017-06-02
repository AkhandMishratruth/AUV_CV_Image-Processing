import cv2
import numpy as np
import time
def centroid_calc(thresh):
    #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #thresh = cv2.Canny(img, 1, 255)
    cv2.imshow('Fuckr', thresh)
    #qwe,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    alpha,contours,hierarchy = cv2.findContours(thresh, 1, 2)    
    cx, cy = -1, -1
    if len(contours) > 0:   
        cnt = contours[0]
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print(M['m00'])
    else:
        print("NF")
    time.sleep(0.1)
    return ([cx, cy])