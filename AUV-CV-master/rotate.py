from contour import centroid_calc

def rotate(img): 
	#start_time = time.time()  #To start Time
	#img = cv2.imread('test.png',0) # Load an color image in binary
     img_y = img.shape[1]/2
     centroid = centroid_calc(img) #Find Centroid of object in (y, x) format
     if centroid == [-1, -1]:
         c = '-1'
     else:
        	obj_y = centroid[0]
        	dist = obj_y - img_y
        	c = '0'
        	if abs(dist) < 20:
        		c = '0'
        	elif dist >= 20:
        		c = 'r'
        	elif dist <= -20:
        		c = 'l' 
	#print("--- %s seconds ---" % (time.time() - start_time))   #To measure Time
     return c






