from __future__ import print_function
import numpy as np
import cv2
import math
feed = cv2.VideoCapture(0)

import cv2.cv as cv
from time import time

boxes = []

def movement(a,b,p_x,p_y,q_x,q_y,r_x,r_y,s_x,s_y,cent_frame_x,cent_frame_y):
    
    t=0
    n=1
    l,r,t,b0,h,v=0,0,0,0,0,0
    b1 = (r_x - q_x + s_x - p_x)/2.0
    a1 = (p_y - q_y + s_y - r_y)/2.0
    cent_door_x = (p_x + q_x + r_x + s_x)/4
    cent_door_y = (p_y + q_y + r_y + s_y)/4
    rot_h = cent_frame_x - cent_door_x
    rot_v = cent_frame_y - cent_door_y
    rot_sign_h = 0
    rot_sign_v = 0
    if rot_h>=0:
        rot_sign_h=1
    else:
        rot_sign_h = -1

    if rot_v>=0:
        rot_sign_v=1
    else:
        rot_sign_v = -1

##    # horizontal shift
##    if b1/a1 < (1-t)*b/a:
##        hr_shift = math.pow(b/a - b1/a1, n)
##        if rot_sign_h == -1:
##            l,r=0,hr_shift
##            print "Move right"
##        elif rot_sign_h == 1:
##            l,r=hr_shift, 0
##            print "Move left"

    # vertical shift
    if abs(rot_v)>abs(rot_h):
        if rot_sign_v == 1:
            print("rotate UP" + str(rot_v),end='    ')
        if rot_sign_v == -1:
            print("rotate Down" + str(rot_v),end='    ')
    else:
        if rot_sign_h== 1:
            print("rotate Left " + str(rot_h),end='    ')
        if rot_sign_h == -1:
            print("rotate Right " + str(rot_h),end='    ')
    if(b1*1.0/a1*1.0)>(b*1.0/a*1.0):
        print("Move up/down "+str((b1*1.0/a1*1.0)/(b*1.0/a*1.0)))
    else:
        print("Move Right/Left "+str((b1*1.0/a1*1.0)/(b*1.0/a*1.0)))
##    if b1/a1 > (1+t)*b/a:
##        vr_shift = math.pow(b1/a1 - b/a, n)
##    if rot_sign_v == 1:
##            t,b=0, vr_shift
##            print "Move down"
##    elif rot_sign_v == -1:
##        print "Move down " + str(math.pow((b1/a1) / (b/a), n))
    #rotation
##    h = rot_sign_h*((1/math.cos(math.pi*rot_h*rot_sign_h/frame_width))-1)
##    v = rot_sign_v*((1/math.cos(math.pi*rot_v*rot_sign_v/frame_width))-1)
##    final= [[l,r],[t,b],[h,v]]
##    return final

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
        'End Mouse Position: ' + str(x) + ', ' + str(y)
        ebox = [x, y]
        boxes.append(ebox)
        print(boxes)
        crop = img[boxes[-2][1]:boxes[-1][1], boxes[-2][0]:boxes[-1][0]]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cv2.imshow('crop', crop)
        k = cv2.waitKey(0)
        if ord('r') == k:
            a = cv2.mean(crop)
            print(a)
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

print(mean_color)
print(std_color)

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
    a = mean_color - 2*std_color
    b = mean_color + 2*std_color
    #mask = cv2.inRange(hsv, cv2.subtract(mean_color, 1*std_color), cv2.add(mean_color, 1*std_color))
    mask = cv2.inRange(hsv, a, b)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal_morph1)
    mask_open_close = cv2.dilate(mask_open, kernal_morph2, iterations=1)
    mask_open_1 = cv2.morphologyEx(mask_open_close, cv2.MORPH_OPEN, kernal_morph11)
    mask_open_close_1 = cv2.dilate(mask_open_1, kernal_morph22, iterations=1)
    res = cv2.bitwise_and(im, im, mask=mask_open_close_1)


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
    final_contour=cv2.convexHull(final_contour)
    cv2.drawContours(res, our_contour, -1, (255,0,0), 3)
    cv2.drawContours(res, final_contour, -1, (255,255,0), 3)
    cv2.imshow('res', res)
    key_cv2 = cv2.waitKey(1)
    if key_cv2 == 13:
        cv2.destroyAllWindows()
        break
    
    if len(final_contour)!=4:
        #print len(final_contour)
        continue
    cv2.drawContours(im, final_contour, -1, (255,0,0), 3)

    cent_frame_y , cent_frame_x=img.shape[:2][0]/2 , img.shape[:2][1]/2
    cent_cont_y=(final_contour[0][0][0]+final_contour[1][0][0]+final_contour[2][0][0]+final_contour[3][0][0])/4
    cent_cont_x=(final_contour[0][0][1]+final_contour[1][0][1]+final_contour[2][0][1]+final_contour[3][0][1])/4
    
##    box=final_contour
##    max1min=[box[0][0][0]+box[0][0][1],box[1][0][0]+box[1][0][1],box[2][0][0]+box[2][0][1],box[3][0][0]+box[3][0][1]]
##    br=box[max1min.index(max(max1min))]
##    tl=box[max1min.index(min(max1min))]
##    tr_mat=[box[0][0]/box[0][1],box[1][0]/box[1][1],box[2][0]/box[2][1],box[3][0]/box[3][1]]
##    tr=box[tr_mat.index(max(tr_mat))]

    box=np.array([[final_contour[0][0][0],final_contour[0][0][1]],[final_contour[1][0][0],final_contour[1][0][1]],[final_contour[2][0][0],final_contour[2][0][1]],[final_contour[3][0][0],final_contour[3][0][1]]])

    left_ind=np.nonzero(box[:,1]<cent_cont_x)
    right_ind=np.nonzero(box[:,1]>cent_cont_x)
    
    left_pts = box[left_ind]
    right_pts = box[right_ind]

    tl_ind = np.nonzero(left_pts[:,0]==min(left_pts[:,0]))[0][0]
    tr_ind = np.nonzero(right_pts[:,0]==min(right_pts[:,0]))[0][0]
    bl_ind = np.nonzero(left_pts[:,0]==max(left_pts[:,0]))[0][0]
    br_ind = np.nonzero(right_pts[:,0]==max(right_pts[:,0]))[0][0]

    tl_pts = tuple(left_pts[tl_ind])
    bl_pts = tuple(left_pts[bl_ind])
    tr_pts = tuple(right_pts[tr_ind])
    br_pts = tuple(right_pts[br_ind])

    tr_pts, bl_pts=bl_pts, tr_pts
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im,'TL',tl_pts, font, 1,(0,0,0),2)
    cv2.putText(im,'TR',tr_pts, font, 1,(0,0,0),2)
    cv2.putText(im,'BL',bl_pts, font,1,(0,0,0),2)
    cv2.putText(im,'BR',br_pts, font, 1,(0,0,0),2)
    
##    leftmost = tuple(final_contour[final_contour[:,:,0].argmin()][0])
##    rightmost = tuple(final_contour[final_contour[:,:,0].argmax()][0])
##    topmost = tuple(final_contour[final_contour[:,:,1].argmin()][0])
##    bottommost = tuple(final_contour[final_contour[:,:,1].argmax()][0])

    p_x,p_y,q_x,q_y,r_x,r_y,s_x,s_y=bl_pts[0],bl_pts[1],tl_pts[0],tl_pts[1],tr_pts[0],tr_pts[1],br_pts[0],br_pts[1]
    cent_frame_x,cent_frame_y=im.shape[:2][1]/2,im.shape[:2][0]/2
    b,a=32,51
    movement(a,b,p_x,p_y,q_x,q_y,r_x,r_y,s_x,s_y,cent_frame_x,cent_frame_y)
    cv2.imshow("img", im)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()

    

    
	
