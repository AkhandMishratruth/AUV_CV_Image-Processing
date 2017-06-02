#testing file
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 05:13:03 2016

@author: sky
"""

import cv2
import numpy as np
img = cv2.imread('Screenshot_2.png')

thresh = cv2.Canny(img, 1, 255)

lines = cv2.HoughLinesP(thresh,1,np.pi/180,10,1,
                        30)
print(lines[1])