#test file
import cv2
import numpy as np
from rotate import rotate

img = cv2.imread('ima.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray2 = gray.copy()
mask = np.zeros(gray.shape,np.uint8)
qwe,contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
max_cnt = contours[0]
max_area = cv2.contourArea(max_cnt)
for cnt in contours:
	area = cv2.contourArea(cnt)
	if(area > max_area):	
		max_area = area
		max_cnt = cnt
		
print(cv2.contourArea(max_cnt))
cv2.drawContours(mask,[max_cnt],0,255,-1)
cv2.imshow('IMG',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(rotate(mask))