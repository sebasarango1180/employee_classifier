import os
import base64
import functions as f
from flask import Flask, render_template, request
from flask_socketio import SocketIO

from keras import backend as K
K.set_image_dim_ordering('th')

app = Flask(__name__)
socketio = SocketIO(app)

model_opt = 'cnn'
cam = f.select_camera('corredor')

'''
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
'''
def ws_to_front(pic):
    socketio.emit('sending_pic', base64.b64encode(pic))

@app.route('/')
def home():
    #render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/train')
def trainer():
    f.train_system(model_opt)


@app.route('/run', methods=['POST'])
def runner():
    cam = request.get_data()
    camera = f.select_camera(cam)
    ws_to_front(f.run_system(model_opt, camera))  # Event: sending_pic.
    return 204


if __name__ == "__main__":

    model_opt = 'cnn'
    cam = f.select_camera('corredor')
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    socketio.run(host='0.0.0.0', port=port, debug=True)
    #optional if we want to run in debugging mode
    #app.run(debug=True)
