import cv2
from datetime import datetime
import os

#Rango: 244 - 254
#IP escalas: 252
#IP conferencias: 253
#IP corredor: 248
#IP cocina: 250
#IP entrada: 251
#IP afuera: 244 - 247,
#IP corporativo: 249
#IP callcenter: 254
#IP bodega: 240


cascade_path = "/home/experimentality/openCV/opencv/data/haarcascades/"
cascade = "haarcascade_frontalface_alt.xml"
cropping_path = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Cropped/"
original = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Original/"
validator = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Validation/"

face = cv2.CascadeClassifier(cascade_path + cascade)

cam_1 = cv2.VideoCapture("rtsp://admin:admin@172.16.1.251:554")

settings = {
    'scaleFactor': 1.5,
    'minNeighbors': 3,
    # 'flags': cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV.HAAR_DO_ROUGH_SEARCH  # OpenCV 2
    'flags': cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,  # OpenCV 3
    'minSize': (30, 30)
}


def crop_face(face, box, n):
    # Box = [x y w h]
    cropped = face[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]  # First Y coords, then X coords.
    #cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV) #Convierto a HSV
    cropped = cv2.resize(cropped, (120, 120))

    crop_name = cropping_path + datetime.now().isoformat() + "_face_" + str(n) + ".png"
    cv2.imwrite(crop_name, cropped)


###########################################################

while True:

    ret, image = cam_1.read()

    detected = face.detectMultiScale(image, **settings)  # Returns list of rectangles in the image.
    print(detected)

    if len(detected):
        n = 1
        for faces in detected:
            crop_face(image, faces, n)

            n += 1


    cv2.imshow('Cropper', image)

    if cv2.waitKey(5) != -1:
        break

    else:
        print "No faces found"

cv2.destroyWindow("Cropper")
