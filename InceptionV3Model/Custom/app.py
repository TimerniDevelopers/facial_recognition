from flask import Flask, render_template, request
from keras.models import load_model
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

model = load_model('model_inception.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():

    img = request.files['img']
    img.save('img.jpg')

    img = image.load_img('img.jpg', target_size=(224, 224))

    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)

    preds = np.argmax(preds, axis=1)
    print(preds)

    if preds == 0:
        preds = "Elias"
    elif preds == 1:
        preds = "Mugdho"
    elif preds == 2:
        preds = "Wahid"
    else:
        preds = "not detected"

    return render_template('prediction.html', data=preds)

if __name__ == '__main__':
    app.run(debug=True)
