import cv2
from datetime import datetime
import os

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


cascade_path = "/home/experimentality/openCV/opencv/data/haarcascades/"
cascade = "haarcascade_frontalface_alt.xml"
cropping_path = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Cropped/"
base_path = "/home/experimentality/Documents/Degree work/Software/employee_classifier/"
original = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Original/"
validator = "/home/experimentality/Documents/Degree work/Software/employee_classifier/Validation/"


#camera = cv2.VideoCapture("rtsp://172.16.1.246:554/")
face = cv2.CascadeClassifier(cascade_path + cascade)

settings = {
    'scaleFactor': 1.1,
    'minNeighbors': 3,
    # 'flags': cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV.HAAR_DO_ROUGH_SEARCH  # OpenCV 2
    'flags': cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,  # OpenCV 3
    'minSize': (40, 40)
}


def select_camera(cam_id):

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

    print("cam_id: {}".format(cam_id))

    cameras = {'corredor': 'rtsp://admin:admin@172.16.1.248:554',
               'corporativo': 'rtsp://admin:admin@172.16.1.249:554',
               'escalas': 'rtsp://admin:admin@172.16.1.252:554'}
    for key, vals in cameras.iteritems():
        print(key)
        if key == cam_id:
            return vals
        else:
            return 'rtsp://admin:admin@172.16.1.248:554'


def crop_face(face, box, n):
    # Box = [x y w h]
    cropped = face[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]  # First Y coords, then X coords.
    #cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2HSV) #Convierto a HSV
    cropped = cv2.resize(cropped, (120, 120))

    crop_name = cropping_path + '.' + datetime.now().isoformat() + "_face_" + str(n) + ".png"
    cv2.imwrite(crop_name, cropped)


def image_to_feature_vector(image):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return np.array(image).flatten()


def pre_processing(data, labels):

    print("[INFO] Estandarizando...")

    # encode the labels, converting them from strings to integers
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # print(labels)

    # data = np.array(data) / 255.0
    # data = np.array(data).astype(float)

    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

    return data, labels, min_max_scaler


def split_data(data, labels):

    # partition the data into training and testing splits, using 80%
    # of the data for training and the remaining 20% for testing
    print("[INFO] Validacion cruzada (80-20)...")

    (trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.2, random_state=42)
    return trainData, testData, trainLabels, testLabels


def feature_extract(trainData, testData, comps):

    print("[INFO] Aplicando PCA...")

    pca = PCA(n_components=comps)
    pca.fit(trainData)
    trainData_pca = pca.transform(trainData)
    testData_pca = pca.transform(testData)

    return trainData_pca, testData_pca, pca


def mlp_model(trainData_pca, trainLabels, testData_pca, testLabels):


    print("[INFO] Entrenando red neuronal...")

    '''parameters = {'alpha': [1e-5, 1e-2, 1, 10, 100], 'hidden_layer_sizes': [(5, 3), (3, 2), (7, 3), (8, 4), (10, 4)],
                  'random_state': [1, 10], 'solver': ('sgd', 'lbfgs')}'''

    parameters = {'alpha': 1, 'hidden_layer_sizes': (8, 3, 2),
                  'random_state': 6, 'solver': 'lbfgs'}
    # (5, 4, 2), (3, 3, 2), (7, 3, 2), (8, 3, 2),

    # [(5, 4, 2), (3, 3, 2), (7, 3, 2), (8, 3, 2), (10, 6, 3)] [(5, 3), (3, 2), (7, 3), (8, 4), (10, 4)]
    mlp = MLPClassifier(max_iter=5000, **parameters)
    #mlp = GridSearchCV(mlp, parameters)  # Find the best classifier based on params.
    mlp.fit(trainData_pca, trainLabels)

    print("Score de clasificacion (entrenamiento):")
    print(mlp.score(trainData_pca, trainLabels))
    print("------------------------------------------------------")

    print("Score de clasificacion (prueba):")
    print(mlp.score(testData_pca, testLabels))
    print("------------------------------------------------------")

    '''print("Mejor estimador neuronal:")
    print(mlp.best_estimator_)
    print("------------------------------------------------------")'''

    y_pred = mlp.predict(testData_pca)  # Predicted.
    print "Predicted labels"
    print(y_pred)
    print "Test labels"
    print(testLabels)

    print("Reporte de clasificacion (prueba):")
    print(classification_report(testLabels, y_pred))
    print("------------------------------------------------------")
    print("Matriz de confusion (prueba):")
    print(confusion_matrix(testLabels, y_pred))
    print("------------------------------------------------------")

    return mlp


def cnn_model(trainData_pca, trainLabels, testData_pca, testLabels):
    trainData_pca = trainData_pca.reshape(trainData_pca.shape[0], 1, 25, 1).astype('float32')
    testData_pca = testData_pca.reshape(testData_pca.shape[0], 1, 25, 1).astype('float32')
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(1, 25, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # 2 classes.
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(trainData_pca, trainLabels, validation_data=(testData_pca, testLabels), epochs=4, batch_size=200)
    # Final evaluation of the model
    scores = model.evaluate(testData_pca, testLabels, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))

    return model


def load_model(modelname):

    return joblib.load(modelname)


def get_cam(camera):

    print(camera)
    camera = cv2.VideoCapture(camera)

    return camera


def get_models(model_opt):
    scaler = load_model(model_opt + '.scaler')
    pca = load_model(model_opt + '.pca')
    model = load_model(model_opt + '.model')

    return scaler, pca, model


def run_system(scaler, pca, model, camera):  # Pass models as arguments.

    '''ret, img = camera.read()

    det = face.detectMultiScale(img, **settings)  # Returns list of rectangles in the image.
    if len(det):

        n = 1
        for faces in det:
            for x, y, w, h in det[-1:]:  # Just in case I'm interested on showing the rectangle.
                imgn = img[y:y+h, x:x+w]
                imgn = cv2.resize(imgn, (120, 120))
                imgn = cv2.cvtColor(imgn, cv2.COLOR_BGR2HSV)  # Convert to HSV

                imgn_fv = image_to_feature_vector(imgn)
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
                    color_rect = (255, 0, 0)
                del y_new
                cv2.rectangle(img, (x, y), (x + w, y + h), color_rect, 2)

            n += 1

    else:
        print('No faces found')

    return img'''

    ret, img = camera.read()

    det = face.detectMultiScale(img, **settings)  # Returns list of rectangles in the image.
    if len(det):

        n = 1
        for faces in det:
            for x, y, w, h in det[-1:]:  # Just in case I'm interested on showing the rectangle.
                imgn = img[y:y+h, x:x+w]
                imgn = cv2.resize(imgn, (120, 120))
                imgn = cv2.cvtColor(imgn, cv2.COLOR_BGR2HSV)  # Convert to HSV

                imgn_fv = image_to_feature_vector(imgn)
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
                    color_rect = (255, 0, 0)
                del y_new
                cv2.rectangle(img, (x, y), (x + w, y + h), color_rect, 2)

            n += 1

    else:
        print('No faces found')

    return img


def train_system(model_opt):

    # initialize the data matrix and labels list
    data = []
    labels = []

    for modelFiles in os.listdir(base_path):
        if os.path.basename(modelFiles).split('.')[-1] == 'model' \
                or os.path.basename(modelFiles).split('.')[-1] == 'pca' \
                or os.path.basename(modelFiles).split('.')[-1] == 'scaler':
            os.remove(modelFiles)

    for imagePath in os.listdir(validator):  # Open cropped pics and turn them into features. (cropping_path)
        # load the image and extract the class label (assuming that our
        # path as the format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(validator + imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        # construct a feature vector raw pixel intensities, then update
        # the data matrix and labels list
        features = image_to_feature_vector(image)
        data.append(features)
        labels.append(label)
        print(imagePath + " -> " + str(label))

    print(labels)
    print("Dimensiones de matriz de datos:")
    print(np.shape(data))
    print("Dimensiones de vector de etiquetas:")
    print(np.shape(labels))

    (data, labels, min_max_scaler) = pre_processing(data, labels)
    print(labels)
    (trainData, testData, trainLabels, testLabels) = split_data(data, labels)
    (trainData_pca, testData_pca, pca) = feature_extract(trainData, testData, comps=25)
    if model_opt == 'mlp':
        mod = mlp_model(trainData_pca, trainLabels, testData_pca, testLabels)
    else:
        mod = cnn_model(trainData_pca, trainLabels, testData_pca, testLabels)

    joblib.dump(mod, model_opt + '.model')
    joblib.dump(pca, model_opt + '.pca')
    joblib.dump(min_max_scaler, model_opt + '.scaler')



