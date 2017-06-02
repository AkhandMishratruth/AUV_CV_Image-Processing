#import cv2.cv as cv
#import cv2
import numpy as np
import time
if not('cv2' in globals() or 'cv2' in locals()):
    import cv2

feed = cv2.VideoCapture(0)
#print("opened: " + str(feed.isOpened()))
#print("grabbed: " + str(feed.grab()))
#time.sleep(1)
#image_size = feed.read()[1]
'''while
cv2.imshow('asd', image_size)
a = cv2.waitKey(10)'''

class canny_att():
    def __init__(self):
        self.t1 = 0
        self.t2 = 0

#canny_obj = canny_att
def toDo(self):
    pass
    '''
    canny_obj.t1 = cv2.getTrackbarPos('t1', 'canny')
    canny_obj.t2 = cv2.getTrackbarPos('t2', 'canny')
    #print("t1: "+str(cv2.getTrackbarPos('t1', 'canny')) + "\tt2: "+str(cv2.getTrackbarPos('t2', 'canny'))+"\tapera: "+str(cv2.getTrackbarPos("apera","canny")))
    print("t1: " + str(canny_obj.t1) + "\tt2: " + str(canny_obj.t2) + "\tapera: " + str(cv2.getTrackbarPos("apera", "canny")))
    '''

cv2.namedWindow('canny')
cv2.createTrackbar('t1', 'canny', 200, 255, toDo)
cv2.createTrackbar('t2', 'canny', 150, 255, toDo)
#cv2.createTrackbar('apera', 'canny', 1, 10, toDo)

#def motion_bw(feed):
while(True):
    a1 = feed.read()[1].astype('uint8')
    #time.sleep(0.4)
    a2 = feed.read()[1].astype('uint8')
    a3 = feed.read()[1].astype('uint8')
    a11 = cv2.GaussianBlur(a1, (3, 3), 3)
    a21 = cv2.GaussianBlur(a2, (3, 3), 3)
    a31 = cv2.GaussianBlur(a3, (3, 3), 3)
    #cv2.imshow('a1', a1)
    #cv2.imshow('a2', a2)
    #cv2.imshow('a3', a3)
    #print(type(abs(a - b).astype('uint8')))
    #c = abs(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)-cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)).astype('uint8')
    c = abs(a1.astype(int)-a2.astype(int)).astype('uint8')
    #d = cv2.cvtColor(abs(cv2.cvtColor(a, cv2.COLOR_BGR2HSV).astype(int)-cv2.cvtColor(b, cv2.COLOR_BGR2HSV).astype(int)).astype('uint8'), cv2.COLOR_HSV2BGR)
    d = abs(a1.astype(int)-2*a2.astype(int)+a3.astype(int)).astype('uint8')
    #c = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)
    cv2.imshow('c', c)
    cv2.imshow('d', d)

    #apera = cv2.getTrackbarPos('apera', 'canny')
    t1 = cv2.getTrackbarPos('t1', "canny")
    t2 = cv2.getTrackbarPos('t2', "canny")
    e = cv2.Canny(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), t1, t2)
    c1 = abs(a11.astype(int) - a21.astype(int)).astype('uint8')
    e1 = cv2.Canny(cv2.cvtColor(c1, cv2.COLOR_BGR2GRAY), t1, t2)
    cv2.imshow('canny', e)
    cv2.imshow("canny1", e1)
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    f1 = cv2.dilate(e1, kernal, iterations=5)
    cv2.imshow("f1", f1)

    if cv2.waitKey(1) == ord('q'):
        feed.release()
        cv2.destroyAllWindows()
        break

    #return e1
