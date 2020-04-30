# -*- coding: utf-8 -*-
"""
Prediction on video using tensorflow model.
"""

# import the necessary packages
from __future__ import print_function
import argparse
import json
from collections import deque
import numpy as np
import cv2
from keras.models import load_model
from keras.applications.resnet import preprocess_input

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--predictions", required=True,
                help=" prediction of 'road' or 'weather' conditions")
ap.add_argument("-m", "--model", required=True,
                help="path to trained serialized model")
ap.add_argument("-i", "--input", required=True,
                help="path to our input video")
ap.add_argument("-l", "--labels", required=True,
                help="path to the labels")
ap.add_argument("-o", "--output", required=True,
                help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=1,
                help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model...")
model = load_model(args["model"])

# load dictionary of labels
print("[INFO] loading labels...")
json_file = open(args['labels'])
label_dict = json.load(json_file)

# initialize the image mean for mean subtraction along with the
# predictions queue
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # clone the output frame, then convert it from BGR to RGB
    # ordering, resize the frame to a fixed 224x224, and then
    # perform mean subtraction
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if args['predictions'] == 'weather':
        frame = cv2.resize(frame, (224, 224)).astype("float32")
    else:
        frame = cv2.resize(frame, (512, 512)).astype("float32")
    frame = preprocess_input(frame)

    # make predictions on the frame and then update the predictions
    # queue
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)

    # perform prediction averaging over the current history of
    # previous predictions
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = label_dict[str(i)]

    # draw the condition on the output frame
    text = "condition: {}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 5)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # write the output frame to disk
    writer.write(output)

    # show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
