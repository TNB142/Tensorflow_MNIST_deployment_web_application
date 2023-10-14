import urllib.request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras_preprocessing.image import load_img

model=load_model("epoch_20.keras")

path_img = "D:\\ACODE\\Kaggle\\model_test_deploy\\static\\upload_image\\img_1.jpg"
# # img=Image.open(path_img)
img = tf.io.read_file(path_img)
img = tf.io.decode_jpeg(img,channels=1)
img = tf.compat.v1.image.resize(img,(28, 28))
print(img.shape)
img = tf.expand_dims(img,0)
print(img.shape)


predictions=model.predict(img,steps=1)
label=np.argmax(predictions)
print(label)
print("end")
# print(model.summary())