import cv2
import numpy as np
def neg(event,x,y,flags,param):
    global x_in, y_in,drawing,tl,br

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing=True
        x_in,y_in=x,y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing:
            tl=(min(x_in,x),min(y_in,y))
            br=(max(x_in,x),max(y_in,y))
            #img[y_init:y,  x_init:x]=255   -	img[y_init:y,x_init:x]
    elif event==cv2.EVENT_LBUTTONUP:
        drawing= False
        tl=(min(x_in,x),min(y_in,y))
        br=(max(x_in,x),max(y_in,y))
        #img[y_in:y, x_in:x]=255-img[y_in:y, x_in:x]
if __name__=='__main__'        :
    drawing = False
    tl,br=(-1,-1),(-1,-1)
    cam=cv2.VideoCapture(0)

    if not cam.isOpened():
        raise IOError("Cam not open")
    cv2.namedWindow('webcam')
    cv2.setMouseCallback('webcam',neg)
    while True:
        ret,frame=cam.read()
        img=cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        (x0,y0),(x1,y1)=tl,br
        img[y0:y1,x0:x1]=255-img[y0:y1,x0:x1]
        cv2.imshow('webcam',img)

        c=cv2.waitKey(1)
        if c==27:
            break
    cap.release()
    cv2.destroyAllWindows()
