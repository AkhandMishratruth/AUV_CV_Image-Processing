# not imporrtant file only for testting
import cv2
from threshold import findangle
import numpy as np
 
img=cv2.imread('q.png',0)

y,contours,hie=cv2.findContours(img,1,2)
cnt=contours[0]
rows,cols=img.shape[:2]
[vx,vy,x,y]=cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
ly=int((-x*vy/vx)+y)
ry=int(((cols-x)*vy/vx)+y)
cv2.line(img,(cols-1,ry),(0,ly),(0,255,0))
cv2.imshow('win',img)
cv2.waitKey(0)
'''

cv2.imshow('win1',img)
cv2.waitKey(2)
box=findangle(img)
cv2.drawContours(img,[box],-1,(0,0,255),4)
cv2.imshow('win',img)
cv2.waitKey(0)
x1=box[0][0]
y1=box[0][1]
x2=box[1][0]
y2=box[1][1]plt.imshow(im1),plt.show()
x3=box[2][0]
y3=box[2][1]
x4=box[3][0]
y4=box[3][1]
l1=(x1-x2)^2+(y1-y2)^2
l2=(x2-x3)^2+(y2-y3)^2
plt.imshow(im1),plt.show()
if(l1>l2):
  x01=x1
  x02=x2
  y01=y1
  y02=y2
else:
  
  x01=x2
  x02=x3
  y01=Line
  y02=y3
slope=abs(x01-x02)/abs(y01-y02)
theta=np.arctan(slope)
theta=(180/3.1415926)*theta
    #print 'theta =' + str(theta[0])
'''
