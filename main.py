import cv2
from datetime import datetime
import os
import functions as f

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

model_opt = 'cnn'
cam = f.select_camera('corredor')
running = True

cascade_path = "/home/experimentality/openCV/opencv/data/haarcascades/"
cascade = "haarcascade_frontalface_alt.xml"
cropping_path = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Cropped/"
original = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Original/"
validator = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Validation/"


camera = cv2.VideoCapture(cam)
face = cv2.CascadeClassifier(cascade_path + cascade)

settings = {
    'scaleFactor': 1.5,
    'minNeighbors': 3,
    # 'flags': cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV.HAAR_DO_ROUGH_SEARCH  # OpenCV 2
    'flags': cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,  # OpenCV 3
    'minSize': (40, 40)
}


f.train_system(model_opt)

scaler = f.load_model(model_opt + '.scaler')
pca = f.load_model(model_opt + '.pca')
model = f.load_model(model_opt + '.model')

while running:

    f.run_system(scaler, pca, model, camera)
    '''
    ret, img = camera.read()

    det = face.detectMultiScale(img, **settings)  # Returns list of rectangles in the image.
    if len(det):

        n = 1
        for faces in det:
            for x, y, w, h in det[-1:]: #Just in case I'm interested on showing the rectangle.
                imgn = img[y:y+h, x:x+w]
                imgn = cv2.resize(imgn, (120, 120))
                imgn = cv2.cvtColor(imgn, cv2.COLOR_BGR2HSV) #Convierto a HSV

                imgn_fv = f.image_to_feature_vector(imgn)
                print("Dimensiones de vector de caracteristicas:")
                print(np.shape(imgn_fv))
                imgn_rs = imgn_fv.reshape(1, -1)
                print("Dimensiones de caracteristicas (reshape):")
                print(np.shape(imgn_rs))
                imgn_ft = scaler.transform(imgn_rs)
                print("Dimensiones de entrada normalizada:")
                print(np.shape(imgn_ft))
                imgn_pca = pca.transform(imgn_ft)
                print("Dimensiones de PCA a entrada:")
                print(np.shape(imgn_pca))
                y_new = model.predict(imgn_pca)
                print("Clasificacion a entrante:")
                print(y_new)

                del imgn

                if y_new[0] == 1:
                    color_rect = (0, 255, 0)
                else:
                    color_rect = (0, 0, 255)
                del y_new
                cv2.rectangle(img, (x, y), (x + w, y + h), color_rect, 2)

            n += 1

    else:
        print('No faces found')

    cv2.imshow('Allowance', img)
    
'''
    if cv2.waitKey(5) != -1:
        break

cv2.destroyWindow("Allowance")
