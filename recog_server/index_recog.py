# start import lib
import os, jwt, json, time, requests
from datetime import datetime, timedelta, date
from flask import Flask, request, redirect, url_for, jsonify, send_file, Response, session
from flask_cors import CORS
import api
#import MyThread
from gevent.pywsgi import WSGIServer
import numpy as np

# define config
SECRECT = 'beetsoft'

# config server
app = Flask(__name__)
CORS(app)

# handle http
@app.route("/")
def index():
    return "Hello REVA!"

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.args.get('token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 403
        try:
            data = jwt.decode(token, SECRECT, algorithms=['HS256'])
        except:
            return jsonify({'message': 'Token is invalid!'}), 403
        return f(*args, **kwargs)
    return decorated

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/api/ekyc/recog_base64', methods=['POST'])
def recog_base64():
    try:
        start = datetime.now()
        request_json = json.loads(json.dumps(request.get_json()))
        data = {
            'image' : request_json['image'],
            'text_bounds': request_json['text_bounds'],
        }

        response = api.batch_process(data)

        end = datetime.now()
        print('[Recognize word] process in', (end-start).total_seconds(), 'seconds', flush=True)
        return jsonify(response)
    except Exception as e:
        print(e, flush=True)
    return jsonify({"result_code":500, "id_check":"","id_logic":"","id_logic_message":"","id_type":"","idconf":"",
                    "server_name":"","server_ver":"1.0", "words":"", "scores":""})

if __name__ == '__main__':

    http_server = WSGIServer(('0.0.0.0', 3010), app)
    http_server.serve_forever()
