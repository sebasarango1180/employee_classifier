import cv2

camera = cv2.VideoCapture("rtsp://admin:admin@172.16.1.254:554")
#Rango: 244 - 254
#IP escalas: 252
#IP corredor: 251

while True:
    ret, img = camera.read()

    cv2.imshow('Check', img)

    if cv2.waitKey(5) != -1:
        break

cv2.destroyWindow("Check")

#Idea: desde web escoger cuarto, y variar con esto simplemente la IP del endpoint.