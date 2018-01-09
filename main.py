#import eventlet
#eventlet.monkey_patch()

import os
from datetime import *
import base64
import functions as f
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO, emit
from multiprocessing import Process

from keras import backend as K
K.set_image_dim_ordering('th')

app = Flask(__name__, static_url_path='')
socketio = SocketIO(app, async_mode="gevent_uwsgi")

print "System ON"
os.system("cp ./sources/logoapc-150x150.png ./sources/cam.png")

model_opt = 'mlp'
(scaler, pca, model) = f.get_models(model_opt)

cam_obj_now = []


# cam = f.select_camera('corredor')


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
    base64.b64encode(pic)
    socketio.emit('stream', pic)


def run_cycle(scaler, pca, model, camera_name):
    print "entre al ciclo infinito"
    try:
        cam_obj = f.get_cam(camera_name)
        lim = datetime.now() + timedelta(minutes=2)
        while datetime.now() <= lim:
            pic = f.run_system(scaler, pca, model, cam_obj)
            img = f.set_image_to_send(pic)
            ws_to_front(img)
    finally:
        print "Liberando camara" + str(camera_name)
        f.release_cam(cam_obj)


'''p_corredor = Process(target= run_cycle, args=(scaler, pca, model, cam_obj_now, f.select_camera('corredor')))
p_corporativo = Process(target= run_cycle, args=(scaler, pca, model, cam_obj_now, f.select_camera('corporativo')))
p_escalas = Process(target= run_cycle, args=(scaler, pca, model, cam_obj_now, f.select_camera('escalas')))'''

@app.route('/')
def home():

    # render out pre-built HTML file right on the index page
    return render_template("index.html", async_mode=socketio.async_mode)


@app.route('/js/<path:path>')  # To include static files.
def send_js(path):
    return send_from_directory('js', path)


@app.route('/sources/<path:path>')  # To include static files.
def send_src(path):
    return send_from_directory('sources', path)


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

    '''if p_corporativo.is_alive():
        p_corporativo.terminate()
        os.system("kill -9 " + str(p_corporativo.pid))
        p_corporativo.join()

        #f.release_cam(f.get_cam(f.select_camera('corporativo')))
    if p_escalas.is_alive():
        p_escalas.terminate()
        os.system("kill -9 " + str(p_escalas.pid))
        p_escalas.join()

    if p_corredor.is_alive():
        p_corredor.terminate()
        os.system("kill -9 " + str(p_corredor.pid))
        p_corredor.join()'''


    if os.environ.get('CAM_PID') is not None:
        pid = os.environ.get('CAM_PID')
        sntnc = "kill -9 " + pid
        print "Matando al " + pid
        os.system(sntnc)

    cam = request.get_data()
    cam = cam.split("&")[0].split('=')[-1]

    if cam == 'corredor':
        '''if p_corporativo.is_alive():
            p_corporativo.terminate()
            os.system("kill -9 " + str(p_corporativo.pid))
            p_corporativo.join()

            #f.release_cam(f.get_cam(f.select_camera('corporativo')))
        if p_escalas.is_alive():
            p_escalas.terminate()
            os.system("kill -9 " + str(p_escalas.pid))
            p_escalas.join()

            #f.release_cam(f.get_cam(f.select_camera('escalas')))'''

        p_corredor = Process(target=run_cycle, args=(scaler, pca, model, f.select_camera('corredor')))
        p_corredor.start()
        print(str(p_corredor.pid))
        os.environ['CAM_PID'] = str(p_corredor.pid)
        #p_corredor.join()

    elif cam == 'corporativo':
        '''if p_corredor.is_alive():
            p_corredor.terminate()
            os.system("kill -9 " + str(p_corredor.pid))
            p_corredor.join()

            #f.release_cam(f.get_cam(f.select_camera('corredor')))
        if p_escalas.is_alive():
            p_escalas.terminate()
            os.system("kill -9 " + str(p_escalas.pid))
            p_escalas.join()

            #f.release_cam(f.get_cam(f.select_camera('escalas')))'''
        p_corporativo = Process(target= run_cycle, args=(scaler, pca, model, f.select_camera('corporativo')))
        p_corporativo.start()
        print(str(p_corporativo.pid))
        os.environ['CAM_PID'] = str(p_corporativo.pid)
        #p_corporativo.join()

    elif cam == 'escalas':
        '''if p_corporativo.is_alive():
            p_corporativo.terminate()
            os.system("kill -9 " + str(p_corporativo.pid))
            p_corporativo.join()

            #f.release_cam(f.get_cam(f.select_camera('corporativo')))
        if p_corredor.is_alive():
            p_corredor.terminate()
            os.system("kill -9 " + str(p_corredor.pid))
            p_corredor.join()

            #f.release_cam(f.get_cam(f.select_camera('corredor')))'''
        p_escalas = Process(target= run_cycle, args=(scaler, pca, model, f.select_camera('escalas')))
        p_escalas.start()
        print(str(p_escalas.pid))
        os.environ['CAM_PID'] = str(p_escalas.pid)
        #p_escalas.join()

    '''camera = f.select_camera(cam)
    (scaler, pca, model) = f.get_models(model_opt)
    camera_obj = f.get_cam(camera)
    #p_runner = Process(target=run_cycle, args=(scaler, pca, model, camera_obj,))
   # p_runner.start()
    #print(p_runner.pid)
    #os.environ['CAM_PID'] = str(p_runner.pid)
    #exportable = "export CAM_PID=$CAM_PID:" + str(p_runner.pid)
    #print(os.environ.get('CAM_PID'))
    #os.system(exportable)'''
    return '', 204


if __name__ == "__main__":

    #cam = f.select_camera('corredor')
    # f.train_system('mlp')

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)
    #app.run(host='0.0.0.0', port=port, debug=True)
