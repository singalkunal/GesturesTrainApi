import tensorflow as tf
import urllib.request
import random
import numpy as np
from PIL import Image
from flask import Flask, request
import pyrebase

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Populate this config dictionary for using firebase
config = {
    "apiKey": "",
    "authDomain": ",
    "databaseURL": "",
    "projectId": "",
    "storageBucket": "",
    "messagingSenderId": "",
    "appId": "",
    "measurementId": ""
}


app = Flask(__name__)

firebase = pyrebase.initialize_app(config)

storage = firebase.storage()

def initialize_base_model():
    global base
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    for layer in base.layers:
        layer.trainable = False

def initialize_data():
    global NUM_CLASSES, IMG_SIZE, MODEL_COUNT, curr_model
    MODEL_COUNT = 0
    NUM_CLASSES = 4
    IMG_SIZE = 224
    initialize_base_model()


def url2img(url):
    resp = urllib.request.urlopen(url)
    img = Image.open(resp)
    # img.thumbnail((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

def shuffle(l1, l2):
    temp = list(zip(l1, l2))
    random.shuffle(temp)
    return zip(*temp)

@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/train', methods=['POST'])
def train():
    if request.method == "POST":
        data = request.get_json()
        urls, labels = shuffle(data['images'], data['labels'])
        EPOCHS = data['epochs']
        global MODEL_COUNT

        if len(labels) == 0 or len(set(labels)) < NUM_CLASSES:
            return ("Provide enough training data.", 403)
        X_train = []
        for url in urls:
            X_train.append(url2img(url))

        X_train = np.asarray(X_train, dtype=np.float32)
        Y_train = np.asarray(tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES), dtype=np.float32)
        print("====>", X_train.shape, Y_train.shape, "<====")

        print("===> Training Model....")
        base_model = base

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Flatten(input_shape=(7,7,1280)),
            tf.keras.layers.Dense(units=256, activation='relu', use_bias=True,
                kernel_initializer=tf.keras.initializers.VarianceScaling()),
            tf.keras.layers.Dense(units=NUM_CLASSES, activation='softmax', use_bias=False,
                kernel_initializer=tf.keras.initializers.VarianceScaling())
        ])

        model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

        BATCH_SIZE = X_train.shape[0] // 3
        if BATCH_SIZE == 0:
            return ("Provide more training examples.", 405)
        history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

        print("===> Model trained <===")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_quant_model = converter.convert()   

        print("===> Model converted <===")

        storage.child("models/model_{}.tflite".format(MODEL_COUNT)).put(tflite_quant_model)
        print("===> Model uploaded <===")
        MODEL_COUNT += 1
        global curr_model
        curr_model = model
        return storage.child("models/model_{}.tflite".format(MODEL_COUNT-1)).get_url(None)

# only for testing purposes
@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        urls, labels = data['images'], data['labels']
        X = []
        for url in urls:
            X.append(url2img(url))

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES), dtype=np.float32)
        preds = curr_model.predict(X)

        print(preds)
        print(Y)
        return "OK"

if __name__ == '__main__':
    app.debug = True
    initialize_data()
    app.run(host='0.0.0.0', port=8080)
