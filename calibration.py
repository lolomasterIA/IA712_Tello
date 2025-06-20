import numpy as np
import glob
import time
import cv2
from djitellopy import Tello

tello = Tello()

# Connexion au drone
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

for i in range(20):
    cv2.imwrite(f"calib/frame_{i:02d}.jpg", frame_read.frame)
    time.sleep(2)

images = glob.glob("calib/*.jpg")

# dimensions des coins internes
pattern_size = (8, 6)         # 9x6 cases → 8x5 coins
square_size = 25.0

objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size           # échelle réelle

obj_points = []   # 3D monde
img_points = []   # 2D image

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                          30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None)

print("RMS reprojection error:", ret)
print("K =", K)
print("dist =", dist.ravel())
