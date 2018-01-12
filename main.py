#import eventlet
#eventlet.monkey_patch()

import os
import psutil
# from datetime import *
# import base64
import functions as f
from flask import Flask, render_template, request, send_from_directory
from flask_socketio import SocketIO
from multiprocessing import Process

from keras import backend as K
K.set_image_dim_ordering('th')

app = Flask(__name__, static_url_path='')
socketio = SocketIO(app, async_mode="gevent_uwsgi")

print "System ON"
os.system("cp ./sources/logoapc-150x150.png ./sources/cam.png")

model_opt = 'mlp'
(scaler, pca, model) = f.get_models(model_opt)

current_pid = [None]

'''def ws_to_front(pic):
    #base64.b64encode(pic)
    socketio.emit('stream', pic)'''


def run_cycle(scaler, pca, model, camera_name):
    print "entre al ciclo infinito"
    try:
        cam_obj = f.get_cam(camera_name)
        #lim = datetime.now() + timedelta(minutes=2)
        #while datetime.now() <= lim:
        while True:
            f.run_system(scaler, pca, model, cam_obj)
            #img = f.set_image_to_send(pic)
            #ws_to_front(pic)
    finally:
        print "Liberando camara: " + str(camera_name)
        f.release_cam(cam_obj)


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

    if psutil.pid_exists(current_pid[0]):
        print "Matando a " + str(current_pid[0])
        psutil.Process(current_pid[0]).kill()
    '''if os.environ.get('CAM_PID') is not None:
        pid = os.environ.get('CAM_PID')
        sntnc = "kill -9 " + pid
        os.system(sntnc)'''
    model_opt = request.get_data()
    model_opt = model_opt.split("&")[0].split('=')[-1]  # Get the real option from the form.
    print(model_opt)
    p_train = Process(target=f.train_system, args=(model_opt,))
    p_train.start()
    current_pid[0] = p_train.pid
    #f.train_system(model_opt)
    return '', 204


@app.route('/run', methods=['POST'])
def runner():

    if psutil.pid_exists(current_pid[0]):
        print "Matando a " + str(current_pid[0])
        psutil.Process(current_pid[0]).kill()

    cam = request.get_data()
    cam = cam.split("&")[0].split('=')[-1]

    if cam == 'corredor':

        p_corredor = Process(target=run_cycle, args=(scaler, pca, model, f.select_camera('corredor')))
        p_corredor.start()
        current_pid[0] = p_corredor.pid
        print(str(p_corredor.pid))
        os.environ['CAM_PID'] = str(p_corredor.pid)

    elif cam == 'corporativo':

        p_corporativo = Process(target=run_cycle, args=(scaler, pca, model, f.select_camera('corporativo')))
        p_corporativo.start()
        current_pid[0] = p_corporativo.pid
        print(str(p_corporativo.pid))
        os.environ['CAM_PID'] = str(p_corporativo.pid)

    elif cam == 'escalas':

        p_escalas = Process(target=run_cycle, args=(scaler, pca, model, f.select_camera('escalas')))
        p_escalas.start()
        current_pid[0] = p_escalas.pid
        print(str(p_escalas.pid))
        os.environ['CAM_PID'] = str(p_escalas.pid)

    return '', 204


if __name__ == "__main__":

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)
    #app.run(host='0.0.0.0', port=port, debug=True)