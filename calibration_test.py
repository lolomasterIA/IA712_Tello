import cv2
import numpy as np
import glob

# par ex. 'calib/frame_00.jpg'
img_path = sorted(glob.glob("calib/*.jpg"))[11]
img = cv2.imread(img_path)
print("Résolution image :", img.shape)

data = np.load("tello_intrinsics_7x9.npz")
K, dist = data["K"], data["dist"]

undist = cv2.undistort(img, K, dist)

cv2.imwrite("calib/debug_raw.jpg", img)
cv2.imwrite("calib/debug_undist.jpg", undist)
