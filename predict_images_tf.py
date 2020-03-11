# USAGE
# python3 predict_images_tf.py --predictions weather --model ./models/tensorflow/WeatherCondi.h5 --input ./input --output ./output_weather --labels ./weather_labels.json
# python3 predict_images_tf.py --predictions road --model ./models/tensorflow/RoadCondi.h5 --input ./input --output ./output_road --labels ./road_labels.json

# import the necessary packages
from keras.models import load_model
from keras.applications.resnet import preprocess_input
import glob
import numpy as np
import json
import argparse
import cv2
import os

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



# loop over images
images = glob.glob(args['input']+'/*.jpg')
print("[INFO] Making predictions on images...")
for image in images:
	# clone the output image, then convert it from BGR to RGB
	# ordering, resize the image to a fixed 224x224, and then
	# perform preprocessing
	img = cv2.imread(image)
	output = img.copy()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	if args['predictions']=='weather':
		img = cv2.resize(img, (224, 224)).astype("float32")
	else:
		img = cv2.resize(img, (512, 512)).astype("float32")
	img = preprocess_input(img)

	# make predictions on the image
	preds = model.predict(np.expand_dims(img, axis=0))[0]
	i = np.argmax(preds)
	label = label_dict[str(i)]

	# draw the condition on the output image
	text = "{}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (255, 0, 0), 5)
	image_name = os.path.basename(image)
	# save the image in the output path
	cv2.imwrite(os.path.join(args['output'], image_name),output) 


print("[INFO] finished!!")
