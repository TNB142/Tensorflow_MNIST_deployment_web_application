#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras_preprocessing.image import load_img



app = Flask(__name__, template_folder='template')

UPLOAD_FOLDER = 'static/upload_image/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

model = tf.keras.models.load_model('model/epoch_20.keras')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path_img = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img = tf.io.read_file(path_img)
        img = tf.io.decode_jpeg(img,channels=1)
        img = tf.compat.v1.image.resize(img,(28, 28))
        img = tf.expand_dims(img,0)

        predictions=model.predict(img,steps=1)
        label=np.argmax(predictions)
        return render_template('index.html', img=path_img, predictions=label)
        # return render_template('index.html',img=path_img)
    return render_template('index.html')


# @app.route('/display/<filename>')
# def display_image(filename):
#     # print('display_image filename: ' + filename)
#     return redirect(url_for("static",filename="upload_image/"+filename), code=301)





if __name__ == "__main__":
    app.run(debug=False)