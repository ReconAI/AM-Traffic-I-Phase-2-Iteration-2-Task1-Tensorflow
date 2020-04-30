# -*- coding: utf-8 -*-
"""
Predict images using tensorflow model.
"""

# import the necessary packages
from __future__ import print_function
import argparse
import glob
import json
import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from keras.models import load_model
from keras.applications.resnet import preprocess_input

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictions", required=True,
                help=" prediction of 'road' or 'weather' conditions")
ap.add_argument("-m", "--model", required=True,
                help="path to trained serialized model")
ap.add_argument("-i", "--input", required=True,
                help="path to our input images")
ap.add_argument("-o", "--output", required=True,
                help="path to our output images")
ap.add_argument("-l", "--labels", required=True,
                help="path to the labels")

args = vars(ap.parse_args())

# load the trained model
print("[INFO] loading model...")
model = load_model(args["model"])

# load dictionary of labels
print("[INFO] loading labels...")
json_file = open(args['labels'])
label_dict = json.load(json_file)

y_true = []
y_pred = []

# loop over images in every category
categories = os.listdir(args['input'])
for cat in categories:
    images = glob.glob(os.path.join(args['input'], cat, '*.jpg'))
    print("[INFO] Making predictions on images...")
    true_label = label_dict[str(cat)]
    for image in images:
        # clone the output image, then convert it from BGR to RGB
        # ordering, resize the image to a fixed 224x224, and then
        # perform preprocessing
        img = cv2.imread(image)
        output = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if args['predictions'] == 'weather':
            img = cv2.resize(img, (224, 224)).astype("float32")
        else:
            img = cv2.resize(img, (512, 512)).astype("float32")
        img = preprocess_input(img)

        # make predictions on the image
        preds = model.predict(np.expand_dims(img, axis=0))[0]
        i = np.argmax(preds)
        label = label_dict[str(i)]
        y_pred.append(label)
        y_true.append(true_label)

        # draw the condition on the output image
        text = "Predicted: {},\n actual: {}".format(label, true_label)
        y0, dy = 50, 37
        for i, line in enumerate(text.split('\n')):
            print(line)
            y = y0 + i*dy
            cv2.putText(output, line, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 5)
        image_name = os.path.basename(image)
        # save the image in the output path
        cv2.imwrite(os.path.join(args['output'], image_name),output) 
print("Classification Report",classification_report(y_true, y_pred, target_names=list(label_dict.values())))
print("Accuracy score", accuracy_score(y_true, y_pred))
print("[INFO] finished!!")
