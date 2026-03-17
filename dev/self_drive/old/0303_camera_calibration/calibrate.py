#%matplotlib inline

""" Notes
Recommended to use at least 20 images to calibrate
Then use test image to check cal
"""


import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

# Read in an image
#img = cv2.imread('test_image.png')
img = cv2.imread('./calibration_wide/GOPR0066.jpg')

# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # prep obj points (0,0,0), (1,0,0)...(7,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) # select only x,y coords

    # grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # find corners
    #ret, corners = cv2.findChessboardCorners(gray,(8,6),None)
    #if ret == True:
    #    imgpoints.append(corners)
    #    objpoints.append(objp)
    #    img = cv2.drawChessboardCorners(img,(8,6),corners,ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints,
                                                       gray.shape[::-1],
                                                       None,
                                                       None)
    
   
    return cv2.undistort(img,mtx,dist,None,mtx)

undistorted = cal_undistort(img, objpoints, imgpoints)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
