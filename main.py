import os
import base64
import functions as f
from flask import Flask, render_template, request
from flask_socketio import SocketIO

from keras import backend as K
K.set_image_dim_ordering('th')

app = Flask(__name__)
socketio = SocketIO(app)

model_opt = 'mlp'
#cam = f.select_camera('corredor')

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
    socketio.emit('stream', base64.b64encode(pic))

@app.route('/')
def home():
    #render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/train', methods=['POST'])
def trainer():
    model_opt = request.get_data()
    model_opt = model_opt.split("&")[0].split('=')[-1]  # Get the real option from the form.
    print(model_opt)
    f.train_system(model_opt)
    return '', 204


@app.route('/run', methods=['POST'])
def runner():
    cam = request.get_data()
    cam = cam.split("&")[0].split('=')[-1]
    camera = f.select_camera(cam)
    (scaler, pca, model) = f.get_models(model_opt)
    camera = f.get_cam(camera)
    for i in range(20):
        ws_to_front(f.run_system(scaler, pca, model, camera))  # Event: sending_pic.
    return '', 204


if __name__ == "__main__":

    #cam = f.select_camera('corredor')
    f.train_system('mlp')
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
    #optional if we want to run in debugging mode
    #app.run(debug=True)
