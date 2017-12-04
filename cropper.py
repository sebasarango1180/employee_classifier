import cv2
from datetime import datetime
import os

cascade_path = "/home/experimentality/openCV/opencv/data/haarcascades/"
cascade = "haarcascade_frontalface_alt.xml"
cropping_path = "/home/experimentality/Documents/Inteligencia/final_inteligencia/Cropped/"
original = "/home/experimentality/Documents/Inteligencia/final_inteligencia/Original/"
validator = "/home/experimentality/Documents/Inteligencia/final_inteligencia/Validation/"

face = cv2.CascadeClassifier(cascade_path + cascade)

settings = {
    'scaleFactor': 1.1,
    'minNeighbors': 3,
    # 'flags': cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV.HAAR_DO_ROUGH_SEARCH  # OpenCV 2
    'flags': cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,  # OpenCV 3
    'minSize': (40, 40)
}


def crop_face(face, box, n, label):
    # Box = [x y w h]
    cropped = face[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]  # First Y coords, then X coords.
    cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV) #Convierto a HSV
    cropped = cv2.resize(cropped, (120, 120))

    crop_name = cropping_path + label + '.' + datetime.now().isoformat() + "_face_" + str(n) + ".png"
    cv2.imwrite(crop_name, cropped)


###########################################################

for imagePath in os.listdir(original): # Take original pics, detect faces and save croppings.
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(original + imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    detected = face.detectMultiScale(image, **settings)  # Returns list of rectangles in the image.

    print(detected)
    if len(detected):
        n = 1
        for faces in detected:
            crop_face(image, faces, n, label)

            n += 1

    else:
        print('No faces found')