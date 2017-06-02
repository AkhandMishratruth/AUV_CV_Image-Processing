from sklearn import cluster
import numpy as np
import cv2

im = cv2.imread("red.jpg")

rows, cols = im.shape[:2]

data=np.zeros([rows*cols, 3],dtype=int)

data[:,0] = im[:,:,0].reshape(rows*cols,1)
data[:,1] = im[:,:,1].reshape(rows*cols,1)
data[:,2] = im[:,:,2].reshape(rows*cols,1)

ms= cluster.MeanShift()
