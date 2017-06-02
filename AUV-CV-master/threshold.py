#file with most of the algos
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors


def threshold(img2,lower,upper):
   def nothing(x):
    pass
   #cv2.namedWindow('img2')
#img3 = cv2.cvtColor(img2,cv2.COLOR_HSV2BGR)
#cv2.imshow('qwe2',img3)
   '''
   img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)



   cv2.createTrackbar('V1','img2',0,255,nothing)
   cv2.createTrackbar('S1','img2',0,255,nothing)
   cv2.createTrackbar('H1','img2',0,179,nothing)

   cv2.createTrackbar('V2','img2',0,255,nothing)
   cv2.createTrackbar('S2','img2',0,255,nothing)
   cv2.createTrackbar('H2','img2',0,179,nothing)
# create switch for ON/OFF functionality
   switch = '0 : OFF \n1 : ON'
#cv2.createTrackbar(switch, 'img2',0,1,nothing)

   while(1):
     cv2.imshow('img2',img2)
    

    # get current positions of four trackbars
     r1 = cv2.getTrackbarPos('V1','img2')
     g1 = cv2.getTrackbarPos('S1','img2')
     b1 = cv2.getTrackbarPos('H1','img2')
     r2 = cv2.getTrackbarPos('V2','img2')
     g2 = cv2.getTrackbarPos('S2','img2')
     b2 = cv2.getTrackbarPos('H2','img2')
     s = cv2.getTrackbarPos(switch,'img2')
    #boundaries = [
	#   [b1 , g1 , r1], [b2 , g2 , r2 ]]
     lower=[b1,g1,r1]
     upper=[b2,g2,r2]
     if s == 0:
         img[:] = 0
     else:     
#for (lower, upper) in boundaries:
	# create NumPy arrays from the boundarie
          lower = np.array(lower, dtype = "uint8")
          upper = np.array(upper, dtype = "uint8")
          #lower=[b1,g1,r1]
          #upper=[b2,g2,r2]	
          mask = cv2.inRange(img3, lower, upper)
          output = cv2.bitwise_and(img3, img3, mask = mask)
          img4 = cv2.cvtColor(output,cv2.COLOR_HSV2BGR)
          kernel = np.ones((3, 3), np.uint8)
          img4 = cv2.morphologyEx(img4, cv2.MORPH_CLOSE, kernel)
          img4 = cv2.morphologyEx(img4, cv2.MORPH_OPEN, kernel)  
          cv2.imshow("img2", np.hstack([img2, img4]))
     k = cv2.waitKey(1) & 0xFF
     if k == 27:
        break 
          '''
   img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)   
   lower = np.array(lower, dtype = "uint8")
   upper = np.array(upper, dtype = "uint8")
          #lower=[b1,g1,r1]
          #upper=[b2,g2,r2]	
   mask = cv2.inRange(img3, lower, upper)
   output = cv2.bitwise_and(img3, img3, mask = mask)
   img_masked = cv2.cvtColor(output,cv2.COLOR_HSV2BGR)   
   kernel = np.ones((3, 3), np.uint8)
   #img4 = cv2.morphologyEx(img4, cv2.MORPH_CLOSE, kernel)
   #img4 = cv2.morphologyEx(img4, cv2.MORPH_OPEN, kernel)    
   masked_gray= cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
   (thresh, img_bin) = cv2.threshold(masked_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
   #cv2.imshow('qwe12',img6)
   im_floodfill=img_bin.copy()
   h, w = img_bin.shape[:2]
   mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
   cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
   im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
   img_binf = img_bin | im_floodfill_inv
     
    # Display images.
   #cv2.imshow("Thresholded Image", im_th)
   #cv2.imshow("Floodfilled Image", im_floodfill)
   #cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
   #cv2.imshow("Foreground", im_out)
   #cv2.waitKey(0)
   return [img_bin,img_masked, img_binf]
   
   

def findangle(img):
    print img.any()
    if(~img.any()):
       return 0
    y,contours,hie=cv2.findContours(img,1,2)
    cnt=contours[0]
    rows,cols=img.shape[:2]
    [vx,vy,x,y]=cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
    slope=float(vy)/float(vx);
    theta=np.arctan(slope)
    theta=90+theta*180/np.pi

    return (90-theta)
    '''edges = cv2.Canny(img,50,150,apertureSize = 3)
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
    #print 'theta =' + str(theta[0])
    return theta'''


def rotation(img):
    
    from rotate import rotate    
    #cv2.imshow('IMG',mask)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return (rotate(img))
   
   
def max_area(gray):
    mask = np.zeros(gray.shape,np.uint8)
    qwe, contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:      
        max_cnt = contours[0]
        max_area = cv2.contourArea(max_cnt)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if(area > max_area):	
                max_area = area
                max_cnt = cnt		
        cv2.drawContours(mask,[max_cnt],0,255,-1)
    return mask
   
   
   
   
   
   
   
   