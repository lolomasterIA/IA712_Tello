import cv2

img = cv2.imread("debug/frame_raw3.jpg")
qr = cv2.QRCodeDetector()

data, pts, truc = qr.detectAndDecode(img)
print("data:", repr(data))
print("pts:", pts)
print("truc:", truc)

if pts is not None:
    pts = pts.astype(int).reshape(-1, 2)
    for j in range(4):
        cv2.line(img, tuple(pts[j]), tuple(pts[(j+1) % 4]), (0, 255, 0), 2)
cv2.imshow("qr", img)
cv2.waitKey(1)
