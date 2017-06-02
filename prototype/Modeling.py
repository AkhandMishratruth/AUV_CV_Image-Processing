import cv2
import numpy as np
import math


# final[[l,r],[t,b],[h,v]]
def movement(a,b,p_x,p_y,q_x,q_y,r_x,r_y,s_x,s_y):
    l,r,t,b,h,v=0,0,0,0,0,0
    b1 = (r_x - q_x + s_x - p_x)/2
    a1 = (p_y - q_y + s_y - r_y)/2
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

    if rot_v>=0:5
        rot_sign_v=1
    else:
        rot_sign_v = -1

    # horizontal shift
    if b1/a1 < (1-t)*b/a:
        hr_shift = math.pow(b/a - b1/a1, n)
        if rot_sign_h = -1:
            l,r=0,hr_shift
        else if rot_sign_h = 1:
            l,r=hr_shift, 0

    # vertical shift
    if b1/a1 > (1+t)*b/a:
        vr_shift = math.pow(b1/a1 - b/a, n)
        if rot_sign_v = 1:
            t,b=0, vr_shift
        else if rot_sign_v = -1:
            t,b=vr_shift,0

    #rotation
    h = rot_sign_h*((1/math.cos(math.pi*rot_h*rot_sign_h/frame_width))-1)
    v = rot_sign_v*((1/math.cos(math.pi*rot_v*rot_sign_v/frame_width))-1)
    final= [[l,r],[t,b],[h,v]]
    return final

        
        
        


    
