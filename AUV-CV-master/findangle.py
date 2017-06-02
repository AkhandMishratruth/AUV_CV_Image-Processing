
import cv2
import numpy as np

img = cv2.imread('testrect.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow('canny', edges)
#cv2.waitKey(0)
lines = cv2.HoughLinesP(edges,1,np.pi/180,10,100)
x1 = lines[0][0]
y1 = lines[1][0]
x2 = lines[2][0]
y2 = lines[3][0]
tanthe = abs(x2 - x1) / abs(y2 - y1)
theta = np.arctan(tanthe)
theta=90-(180/3.1415926)*theta
print 'theta =' + str(theta[0])