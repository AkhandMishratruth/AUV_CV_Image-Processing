#find largest contour algo
import numpy as np
import cv2


gray=cv2.imread('ima.png',0)
(thresh, img3) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

image, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(img3.shape, np.uint8)
largest_areas = sorted(contours, key=cv2.contourArea)

cv2.drawContours(mask, [largest_areas[1]], 0, (255,255,255,255), -1)
#removed = cv2.subtract(gray, mask)
#cv2.imshow('qwer',gray)
img2=img3-mask
cv2.imshow('qwer',mask)
cv2.waitKey(0)
cv2.destroyAllWindows