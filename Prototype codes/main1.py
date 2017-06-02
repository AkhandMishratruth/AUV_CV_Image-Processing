import numpy as np
import cv2

feed = cv2.VideoCapture(0)

'''
lower_color = np.array([0,0,0])
mean_color = np.array([110, 70, 23])
std_color = np.array([4 + 10, 13 + 30, 12 + 50])
upper_color = np.array([20, 20, 20])
'''

########################################################################################
import cv2.cv as cv

from time import time

boxes = []

def on_mouse(event, x, y, flags, params):
    # global img
    t = time()

    if event == cv.CV_EVENT_LBUTTONDOWN:
        print
        'Start Mouse Position: ' + str(x) + ', ' + str(y)
        sbox = [x, y]
        boxes.append(sbox)
        # print count
        # print sbox

    elif event == cv.CV_EVENT_LBUTTONUP:
        print
        'End Mouse Position: ' + str(x) + ', ' + str(y)
        ebox = [x, y]
        boxes.append(ebox)
        print boxes
        crop = img[boxes[-2][1]:boxes[-1][1], boxes[-2][0]:boxes[-1][0]]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cv2.imshow('crop', crop)
        k = cv2.waitKey(0)
        if ord('r') == k:
            a = cv2.mean(crop)
            print a
            mean_color = [0, 0, 0]
            mean_color[0] = (a[0])
            mean_color[1] = (a[1])
            mean_color[2] = (a[2])
            std_color = [0, 0, 0]
            std_color[0] = int(crop[:, :, 0].std() + 5)
            std_color[1] = int(crop[:, :, 1].std() + 10)
            std_color[2] = int(crop[:, :, 2].std() + 10)
            np.save('mean_save', mean_color)
            np.save("std_save", std_color)

count = 0
while 1:
    count += 1
    img = feed.read()[1]
    # img = cv2.blur(img, (3,3))
    img = cv2.resize(img, None, fx=1, fy=1,interpolation=cv2.INTER_AREA)

    cv2.namedWindow('real image')
    cv.SetMouseCallback('real image', on_mouse, 0)
    cv2.imshow('real image', img)

    if cv2.waitKey(33) == 27:
        cv2.destroyAllWindows()
        break
    elif count >= 50:
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            break
        count = 0

#################################################################################################3

mean_color = np.load("mean_save.npy")
std_color = np.load("std_save.npy")

print mean_color
print std_color

minLineLength = 120
maxLineGap = 1

minThresh = 200
maxThresh = 220

kernal_morph1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
kernal_morph11 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

kernal_morph2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
kernal_morph22 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))


while(1):
    _, im = feed.read()
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    a = mean_color - 3*std_color
    b = mean_color + 3*std_color
    #mask = cv2.inRange(hsv, cv2.subtract(mean_color, 1*std_color), cv2.add(mean_color, 1*std_color))
    mask = cv2.inRange(hsv, a, b)

    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal_morph1)
    mask_open_close = cv2.dilate(mask_open, kernal_morph2, iterations=1)
    mask_open_1 = cv2.morphologyEx(mask_open_close, cv2.MORPH_OPEN, kernal_morph11)
    mask_open_close_1 = cv2.dilate(mask_open_1, kernal_morph22, iterations=1)
    res = cv2.bitwise_and(im, im, mask=mask_open_close_1)

    # print cv2.cvtColor(res, cv2.COLOR_BGR2GRAY).dtype
    cv2.imshow('open', mask_open)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
        break
    cv2.imshow('close', mask_open_close)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
        break
    cv2.imshow('open1', mask_open_1)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
        break
    cv2.imshow('close1', mask_open_close_1)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
        break

    #edge_canny = cv2.Canny(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), minThresh, maxThresh, apertureSize=3)
    #edge_canny = cv2.morphologyEx(edge_canny,cv2.MORPH_OPEN,kernal_morph)

    cv2.imshow('res', res)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
        break

    #lines = cv2.HoughLines(edge_canny,1,np.pi/180,120)
    #for rho,theta in lines[0]:
       # a = np.cos(theta)
      #  b = np.sin(theta)
        #x0 = a*rho
        #y0 = b*rho
        #x1 = int(x0 + 1000*(-b))
       # y1 = int(y0 + 1000*(a))
       # x2 = int(x0 - 1000*(-b))
       # y2 = int(y0 - 1000*(a))
       # cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)

    '''
    lines = cv2.HoughLinesP(edge_canny, 1, np.pi/180, 100, minLineLength, maxLineGap)
    if lines != None:
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('asd',im)
    key_cv2 = cv2.waitKey(100)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
        break
    '''

    contours,_ = cv2.findContours(mask_open_close_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print contours

    perimeter = []
    area = []
    for i in contours:
        area.append(cv2.contourArea(i))
        perimeter.append(cv2.arcLength(i, True))
    area_peri_ratio = []
    for i in range(0, area.__len__()-1, 1):
        area_peri_ratio.append(area[i]/perimeter[i])
    #print str(area_peri_ratio)
    cv2.waitKey(1)
