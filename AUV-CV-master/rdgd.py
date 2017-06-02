#Hough line algo
#   -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 04:17:45 2016

@author: skyy
"""

#img
thresh = cv2.Canny(img, 1, 255)
img_x = thresh.shape[1]/2
lines = cv2.HoughLinesP(thresh,1,np.pi/180,100,0,1000)
c = '-1'
if(len(lines) > 0):
    x1 = lines[0][1]
    dist = x1 - img_x
    if abs(dist) < 20:
        c = '0'
    elif dist >= 20:
        c = 'r'
    elif dist <= -20:
        c = 'l'

