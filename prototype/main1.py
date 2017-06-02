import numpy as np
import cv2

feed = cv2.VideoCapture(0)

import cv2.cv as cv
from time import time

boxes = []
def small(a, b, cent):
    if a[0] - cent[0] >= 0 > b[0] - cent[0]:
        return True
    if a[0] - cent[0] < 0 <= b[0] - cent[0]:
        return False
    if a[0] - cent[0] == 0 and b[0] - cent[0] == 0:
        if a[1] - cent[1] >= 0 or b[1] - cent[1] >= 0:
            return a[1] > b[1]
        return b[1] > a[1]

    det = (a[0] - cent[0]) * (b[1] - cent[1]) - (b[0] - cent[0]) * (a[1] - cent[1])
    if det < 0:
        return True
    if det > 0:
        return False

    d1 = (a[0] - cent[0]) * (a[0] - cent[0]) + (a[1] - cent[1]) * (a[1] - cent[1])
    d2 = (b[0] - cent[0]) * (b[0] - cent[0]) + (b[1] - cent[1]) * (b[1] - cent[1])
    return d1 > d2

# .........................................................................

def sort(l, c):
    for i in range(len(l)-1):
        j = i+1
        while j < len(l):
            if small(l[i][0], l[j][0], c):
                t = tuple(l[j][0])
                l[j][0] = l[i][0]
                l[i][0] = list(t)
            j += 1
    return l

def on_mouse(event, x, y, flags, params):
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

    cv2.imshow('res', res)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
        break

    cnts,_ = cv2.findContours(mask_open_close_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cntsALL,_ = cv2.findContours(mask_open_close_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print contours
    all_cnts_area=[]
    for c in cnts:
        a=cv2.contourArea(c)
        all_cnts_area.append(a)
    all_cnts_area=np.array(all_cnts_area)
    if (all_cnts_area.size==0):
        continue
    maxArea_index=np.nonzero(all_cnts_area==max(all_cnts_area))[0][0]
    our_contour=cnts[maxArea_index]
    #our_contour1=cnts[maxArea_index]
    epsilon= 0.1*cv2.arcLength(our_contour, True)
    final_contour=cv2.approxPolyDP(our_contour, epsilon , True)
    if len(final_contour)!=4:
        continue
    cv2.drawContours(im, final_contour, -1, (0,255,0), 3)
    cv2.imshow("img", im)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
    
	
