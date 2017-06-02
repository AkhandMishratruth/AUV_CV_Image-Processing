import numpy as np
import cv2

im = cv2.imread('test.jpg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
area=[]
for c in contours:
	a = cv2.contourArea(c)
	area.append(a)
area=np.array(area)
print np.nonzero(area==max(area))[0][0]