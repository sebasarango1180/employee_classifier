import os
import base64
import functions as f
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO
from multiprocessing import Process

from keras import backend as K
K.set_image_dim_ordering('th')

app = Flask(__name__, static_url_path='')
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


def run_cycle(scaler, pca, model, camera):
    print "entre al ciclo infinito"
    while True:
        pic = f.run_system(scaler, pca, model, camera)
        if pic is not None:
            ws_to_front(pic)  # Event: sending_pic.
            print "0i3 cY"


def ws_to_front(pic):
    print "Enviando img"
    socketio.emit('stream', pic)


#def ws_to_front(pic):


@app.route('/')
def home():

    # render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/js/<path:path>')  # To include static files.
def send_js(path):
    return send_from_directory('js', path)


@app.route('/train', methods=['POST'])
def trainer():

    if os.environ.get('CAM_PID') is not None:
        pid = os.environ.get('CAM_PID')
        sntnc = "kill -9 " + pid
        os.system(sntnc)
    model_opt = request.get_data()
    model_opt = model_opt.split("&")[0].split('=')[-1]  # Get the real option from the form.
    print(model_opt)
    f.train_system(model_opt)
    return '', 204


@app.route('/run', methods=['POST'])
def runner():

    if os.environ.get('CAM_PID') is not None:
        pid = os.environ.get('CAM_PID')
        sntnc = "kill -9 " + pid
        print "Matando al " + pid
        os.system(sntnc)
    cam = request.get_data()
    cam = cam.split("&")[0].split('=')[-1]
    camera = f.select_camera(cam)
    (scaler, pca, model) = f.get_models(model_opt)
    camera = f.get_cam(camera)
    p_runner = Process(target=run_cycle, args=(scaler, pca, model, camera,))
    p_runner.start()
    print(p_runner.pid)
    os.environ['CAM_PID'] = str(p_runner.pid)
    exportable = "export CAM_PID=$CAM_PID:" + str(p_runner.pid)
    print(os.environ.get('CAM_PID'))
    os.system(exportable)
    return '', 204


if __name__ == "__main__":

    #cam = f.select_camera('corredor')
    # f.train_system('mlp')

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
