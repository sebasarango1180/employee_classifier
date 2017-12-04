###STUDENT CLASSIFICATOR

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import cv2
import os

cascade_path = "/home/experimentality/openCV/opencv/data/haarcascades/"
cascade = "haarcascade_frontalface_alt.xml"
cropping_path = "/home/experimentality/Documents/Inteligencia/final_inteligencia/Cropped/"
original = "/home/experimentality/Documents/Inteligencia/final_inteligencia/Original/"
validator = "/home/experimentality/Documents/Inteligencia/final_inteligencia/Validation/"
camera = cv2.VideoCapture(0)

face = cv2.CascadeClassifier(cascade_path + cascade)

settings = {
    'scaleFactor': 1.1,
    'minNeighbors': 3,
    # 'flags': cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV.HAAR_DO_ROUGH_SEARCH  # OpenCV 2
    'flags': cv2.CASCADE_FIND_BIGGEST_OBJECT | cv2.CASCADE_DO_ROUGH_SEARCH,  # OpenCV 3
    'minSize': (40, 40)
}


def image_to_feature_vector(image):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return np.array(image).flatten()


###########################################################

# initialize the data matrix and labels list
data = []
labels = []

for imagePath in os.listdir(cropping_path):  # Open cropped pics and turn them into features.
    # load the image and extract the class label (assuming that our
    # path as the format: /path/to/dataset/{class}.{image_num}.jpg
    image = cv2.imread(cropping_path + imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    # construct a feature vector raw pixel intensities, then update
    # the data matrix and labels list
    features = image_to_feature_vector(image)
    data.append(features)
    labels.append(label)
    print(imagePath + " -> " + str(label))

print("Dimensiones de matriz de datos:")
print(np.shape(data))
print("Dimensiones de vector de etiquetas:")
print(np.shape(labels))

print("[INFO] Normalizando...")

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# print(labels)

# data = np.array(data) / 255.0
# data = np.array(data).astype(float)

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)

# partition the data into training and testing splits, using 80%
# of the data for training and the remaining 20% for testing
print("[INFO] Validacion cruzada (80-20)...")

(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.2, random_state=42)

print("[INFO] Aplicando PCA...")

pca = PCA(n_components=0.85)
pca.fit(trainData)
trainData_pca = pca.transform(trainData)
testData_pca = pca.transform(testData)

print("[INFO] Entrenando red neuronal...")

parameters = {'alpha': [1e-5, 1e-2, 1, 10, 100], 'hidden_layer_sizes': [(5, 3), (3, 2), (7, 3), (8, 4), (10, 4)],
              'random_state': [1, 10], 'solver': ('sgd', 'lbfgs')}
#(5, 4, 2), (3, 3, 2), (7, 3, 2), (8, 3, 2),

#[(5, 4, 2), (3, 3, 2), (7, 3, 2), (8, 3, 2), (10, 6, 3)] [(5, 3), (3, 2), (7, 3), (8, 4), (10, 4)]
mlp = MLPClassifier(max_iter=500000)
mlp = GridSearchCV(mlp, parameters)  # Find the best classifier based on params.
mlp.fit(trainData_pca, trainLabels)

print("Score de clasificacion (entrenamiento):")
print(mlp.score(trainData_pca, trainLabels))
print("------------------------------------------------------")

print("Score de clasificacion (prueba):")
print(mlp.score(testData_pca, testLabels))
print("------------------------------------------------------")

print("Mejor estimador neuronal:")
print(mlp.best_estimator_)
print("------------------------------------------------------")

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

# print("[INFO] Validando con muestras externas...")

# for imagePath in os.listdir(validator):
#    incoming = np.array(cv2.imread(validator + imagePath)).astype(float)
#    print(validator + imagePath)
#    print(mlp.predict(incoming))

# ===================================================================


while True:
    ret, img = camera.read()

    #img = cv2.imread("/home/experimentality/Documents/Inteligencia/final_inteligencia/Validation/grito2_alejo.jpg")

    det = face.detectMultiScale(img, **settings)  # Returns list of rectangles in the image.
    if len(det):

        n = 1
        for faces in det:
            for x, y, w, h in det[-1:]: #Just in case I'm interested on showing the rectangle.
                imgn = img[y:y+h, x:x+w]
                imgn = cv2.resize(imgn, (120, 120))
                imgn = cv2.cvtColor(imgn,cv2.COLOR_BGR2HSV) #Convierto a HSV

                imgn_fv = image_to_feature_vector(imgn)
                print("Dimensiones de vector de caracteristicas:")
                print(np.shape(imgn_fv))
                imgn_rs = imgn_fv.reshape(1, -1)
                print("Dimensiones de caracteristicas (reshape):")
                print(np.shape(imgn_rs))
                imgn_ft = min_max_scaler.transform(imgn_rs)
                print("Dimensiones de entrada normalizada:")
                print(np.shape(imgn_ft))
                imgn_pca = pca.transform(imgn_ft)
                print("Dimensiones de PCA a entrada:")
                print(np.shape(imgn_pca))
                y_new = mlp.predict(imgn_pca)
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

    if cv2.waitKey(5) != -1:
        break

cv2.destroyWindow("Allowance")

'''
#img = cv2.imread("/home/experimentality/Documents/Inteligencia/final_inteligencia/Validation/Serio_alejo3.jpg")
#img = cv2.imread("/home/experimentality/Documents/Inteligencia/final_inteligencia/Validation/enojo2_alejo.jpg")
v_files = os.listdir(validator)

for fil in v_files:

    orig_name = fil
    img = cv2.imread(validator + fil)

    det = face.detectMultiScale(img, **settings)  # Returns list of rectangles in the image.
    if len(det):

        n = 1
        for faces in det:
            for x, y, w, h in det[-1:]: #Just in case I'm interested on showing the rectangle.
                imgn = img[y:y+h, x:x+w]
                imgn = cv2.cvtColor(imgn,cv2.COLOR_BGR2HSV) #Convierto a HSV
                imgn = cv2.resize(imgn, (120, 120))


                imgn_fv = image_to_feature_vector(imgn)
                #print("Dimensiones de vector de caracteristicas:")
                #print(np.shape(imgn_fv))
                imgn_rs = imgn_fv.reshape(1, -1)
                #print("Dimensiones de caracteristicas (reshape):")
                #print(np.shape(imgn_rs))
                imgn_ft = min_max_scaler.transform(imgn_rs)
                #print("Dimensiones de entrada normalizada:")
                #print(np.shape(imgn_ft))
                imgn_pca = pca.transform(imgn_ft)
                #print("Dimensiones de PCA a entrada:")
                #print(np.shape(imgn_pca))
                y_new = mlp.predict(imgn_pca)
                #print("Clasificacion a entrante:")
                print(orig_name + " -> " + str(y_new))

                del y_new

            n += 1

    else:
        print('No faces found')
'''