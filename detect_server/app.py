from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import urllib.request
import os, shutil
from werkzeug.utils import secure_filename
from CRAFT import contour
import cv2
import requests
import base64
import json
import io
from gtts import gTTS
from playsound import playsound
   
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
AUDIO_FOLDER = 'static/audio/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        # -------------test sending image------
        test_url ='http://localhost:3010/api/ekyc/recog_base64'

        # prepare headers for http request
        headers = {'Content-Type': 'application/json'}

        with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as img_file:
            image_string = base64.b64encode(img_file.read())
        
        sortedContours = contour.getContour(UPLOAD_FOLDER, filename)
        jsonRequest = { 
            'image': image_string.decode('utf-8'),
            'text_bounds': sortedContours,
            'height': 10,
            'width': 20
         }
        response = requests.post(test_url, data=json.dumps(jsonRequest), headers=headers)
        textToRead = json.loads(response.text)['words']
        tts = gTTS(text=textToRead, lang='vi')
        # stringmp3 = ''.join([filename, 'convertedText.mp3'])
        
        tts.save(os.path.join(app.config['AUDIO_FOLDER'], 'convertedText.mp3'))

        # for playing
        playsound(os.path.join(app.config['AUDIO_FOLDER'], 'convertedText.mp3'))

        for filename in os.listdir('static/audio'):
            file_path = os.path.join('static/audio', filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        flash('Image successfully uploaded and displayed below')        
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()
