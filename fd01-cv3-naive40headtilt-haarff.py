###Face detector: here faces are detected, cropped and saved for posterior training of the recognition system.

from math import sin, cos, radians

import cv2
from PIL import Image
from datetime import datetime

cascade_path = "/home/experimentality/openCV/opencv/data/haarcascades/"
cascade = "haarcascade_frontalface_alt.xml"
cascade1 = "haarcascade_frontalface_alt2.xml"
cropping_path = "/home/experimentality/Documents/Degree work/Software/FaceDetection/Cropped/"
camera = cv2.VideoCapture(1)
face = cv2.CascadeClassifier(cascade_path + cascade)

settings = {
    'scaleFactor': 1.1,
    'minNeighbors': 3,
    # 'flags': cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV.HAAR_DO_ROUGH_SEARCH  # OpenCV 2
    'flags': cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,  # OpenCV 3
    'minSize': (30, 30)
}


def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result


def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1] * 0.4
    y = pos[1] - img.shape[0] * 0.4
    newx = x * cos(radians(angle)) + y * sin(radians(angle)) + img.shape[1] * 0.4
    newy = -x * sin(radians(angle)) + y * cos(radians(angle)) + img.shape[0] * 0.4
    return int(newx), int(newy), pos[2], pos[3]


def crop_face(face, box, n):
    # Box = [x y w h]
    cropped = face[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]  # First Y coords, then X coords.
    cropped = cv2.resize(cropped, (120, 120))
    #cv2.imshow("cropped", cropped)
    crop_name = cropping_path + datetime.now().isoformat() + "_face_" + str(n) + ".png"
    cv2.imwrite(crop_name, cropped)
    #cv2.waitKey(0)


###########################################################

while True:
    ret, img = camera.read()

    for angle in [0, -25, 25]:
        rimg = rotate_image(img, angle)
        detected = face.detectMultiScale(rimg, **settings)  # Returns list of rectangles in the image.
        if len(detected):
            n = 1
            for faces in detected:
                print(detected)
                print(faces)  # Prints coords [x y w h] for every face detected.
                detected = [rotate_point(detected[-1], img, -angle)]
                crop_face(rimg, faces, n)
                #for x, y, w, h in detected[-1:]: #Just in case I'm interested on showing the rectangle.
                    #print(len(detected), x, y, x + w, y + h)
                    #crop_face(rimg, faces, n)
                    #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

                n += 1

        else:
            print('No faces found')

    cv2.imshow('facedetect', img)

    if cv2.waitKey(5) != -1:
        break

cv2.destroyWindow("facedetect")
#cv2.destroyWindow("cropped")
