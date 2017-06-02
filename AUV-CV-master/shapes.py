# custom algo not of much use
import numpy as np
import cv2
# Create a black image
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# Draw a diagonal blue line with thickness of 5 px
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
img = cv2.rectangle(img,(200,0),(510,128),(0,255,0),3)
cv2.imshow('image',img)
  
cv2.waitKey(0) 
    
  
cv2.startWindowThread()
cv2.destroyAllWindows()
