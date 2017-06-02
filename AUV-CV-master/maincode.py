# start of code
import cv2
import numpy as np
import urllib
import os
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from threshold import threshold, findangle, rotation, max_area
low=[0,0,0]
up=[0,0,0]

def nothing(x):
    pass
#img = cv2.imread('ink.png')

stream=urllib.urlopen('http://10.42.0.140:8080/video?.mjpeg') #change the address to your cam's ip
bytes=''
while True:
    bytes += stream.read(1024)
    # 0xff 0xd8 is the starting of the jpeg frame
    a = bytes.find('\xff\xd8')
    # 0xff 0xd9 is the end of the jpeg frame
    b = bytes.find('\xff\xd9')
    # Taking the jpeg image as byte stream
    if a!=-1 and b!=-1:
        os.system ( 'clear' )
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        # Decoding the byte stream to cv2 readable matrix format
        thresh_image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
        # Display
        k = cv2.waitKey(1) & 0xFF
        cv2.imshow('Set Threshold',thresh_image)
        if k == 27:
           cv2.destroyAllWindows()
           stream.close()
           break 


         
cv2.namedWindow('Threshold_image')
cv2.createTrackbar('V1','Threshold_image',0,255,nothing)
cv2.createTrackbar('S1','Threshold_image',0,255,nothing)
cv2.createTrackbar('H1','Threshold_image',0,179,nothing)

cv2.createTrackbar('V2','Threshold_image',0,255,nothing)
cv2.createTrackbar('S2','Threshold_image',0,255,nothing)
cv2.createTrackbar('H2','Threshold_image',0,179,nothing)
# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
#cv2.createTrackbar(switch, 'img2',0,1,nothing)




while(1):
     cv2.imshow('Threshold_image',thresh_image)

    # get current positions of four trackbars
     v1 = cv2.getTrackbarPos('V1','Threshold_image')
     s1 = cv2.getTrackbarPos('S1','Threshold_image')
     h1 = cv2.getTrackbarPos('H1','Threshold_image')
     v2 = cv2.getTrackbarPos('V2','Threshold_image')
     s2 = cv2.getTrackbarPos('S2','Threshold_image')
     h2 = cv2.getTrackbarPos('H2','Threshold_image')
     s = cv2.getTrackbarPos(switch,'Threshold_image')
    #boundaries = [
	#   [b1 , g1 , r1], [b2 , g2 , r2 ]]
     
     if s == 0:
         img[:] = 0
     else:     
          lower=[h1,s1,v1]
          upper=[h2,s2,v2] 
          global low     
          low = lower
          global up     
          up = upper
          
          (img_bin,img_masked,img_binf)=threshold(thresh_image,lower,upper)
          cv2.imshow("Threshold_image", np.hstack([thresh_image, img_masked]))
     k = cv2.waitKey(1) & 0xFF
     if k == 27:
        cv2.destroyAllWindows()
        break  
   
print 'hello'
#(img6,img4,imgbin)=threshold(img4)
while(True):
 stream=urllib.urlopen('http://10.42.0.140:8080/video?.mjpeg')
 bytes =''
 for i in range(1,150):
  #print 'hello2'  
  
  bytes += stream.read(1024)
    # 0xff 0xd8 is the starting of the jpeg frame
  a = bytes.find('\xff\xd8')
    # 0xff 0xd9 is the end of the jpeg frame
  b = bytes.find('\xff\xd9')
    # Taking the jpeg image as byte stream
  if a!=-1 and b!=-1:
        os.system ( 'clear' )
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        # Decoding the byte stream to cv2 readable matrix format
        img_cam = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
        hist,bins = np.histogram(img_cam.flatten(),256,[0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf *hist.max()/ cdf.max() # this line not necessary.

#plt.plot(cdf_normalized, color = 'b')
#plt.hist(img.flatten(),256,[0,256], color = 'r')
#plt.xlim([0,256])
#plt.legend(('cdf','histogram'), loc = 'upper left')
#plt.show()

        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        img_histo = cdf[img_cam]
#cv2.imshow("qwe",img2)

        (img_bin,img_masked,img_binf)=threshold(img_histo,low,up)
        img_binf=max_area(img_binf)
        cv2.imshow('qwerty12',img_binf)
         
        #theta=findangle(imgbin)
        
        
        print(rotation(img_binf))
        k = cv2.waitKey(1) & 0xFF
        #theta=findangle(img_binf)
        #print theta        
        if k == 27:
          cv2.destroyAllWindows()        
          break
 stream.close()
cv2.imshow('qwerty',img_masked)
  

 
 
 
 
 
 
 


'''import matplotlib.colors as colors

def histeq(im,nbr_bins=256):

    # get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 1 * cdf / cdf[-1] # normalize (value component max is 1.0)

    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape), cdf

#equalise the intensity component
img3[:,:,2], cdf = histeq(img3[:,:,2])

#then convert back to RGB space
img4 = colors.hsv_to_rgb(img3)

#img4 = cv2.cvtColor(img3,cv2.COLOR_HSV2BGR)
cv2.imshow('wer',img4)
'''
