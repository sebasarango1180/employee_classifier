import cv2

#camera = cv2.VideoCapture("rtsp://admin:admin@172.16.1.227:554")
#camera = cv2.VideoCapture("rtsp://admin:Pcmayorista01@172.16.1.243:554")
#Rango: 244 - 254
#IP conferencias: 253
#IP escalas: 252
#IP entrada: 251
#IP cocina: 250
#IP corporativo: 249
#IP corredor: 248
#IP afuera: 247
#IP dementores: 236

while True:
    ret, img = camera.read()

    cv2.imshow('Check', img)

    if cv2.waitKey(5) != -1:
        break

cv2.destroyWindow("Check")

#Idea: desde web escoger cuarto, y variar con esto simplemente la IP del endpoint.