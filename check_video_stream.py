import cv2

camera = cv2.VideoCapture("rtsp://admin:admin@172.16.1.246:554")

while True:
    ret, img = camera.read()

    cv2.imshow('Check', img)

    if cv2.waitKey(5) != -1:
        break

cv2.destroyWindow("Check")