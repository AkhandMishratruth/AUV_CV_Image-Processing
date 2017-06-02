import cv2
import numpy as np
import time
start_time = time.time()
img = cv2.imread('circle.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

print "centroid " +str(cx)+" " +str(cy)
print("--- %s seconds ---" % (time.time() - start_time))
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


