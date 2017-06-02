import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('arrow.png',0)
ret,thresh = cv2.threshold(img,127,255,0)
alpha,contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print M

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

print "centroid " +str(cx)+" " +str(cy)

im1 = cv2.imread('arrow.png')
crop_img = im1[0:20, 20:50] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
cv2.circle(im1,(cx,cy), 5, (255,0,255), -1)
im1[173,92] = [0,0,0]
cv2.imshow('im1', im1)
#plt.imshow(im1),plt.show()
cv2.waitKey(0)