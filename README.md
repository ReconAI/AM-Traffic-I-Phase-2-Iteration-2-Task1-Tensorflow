# AM-Traffic-I-Phase-2-Iteration-2-Task1-Tensorflow

Prepare and develop end-to-end pipeline for a weather conditions classification light-weight neural network. 

As an input this model should take a video sequence from CCTV camera; As an output model should classify weather conditions (Clear, Rain, Snow). 

Network should be light enough to run in realtime on a Jetson Nano device.

-------------------------------------------------------------------------------------------------------------------------------

# Data
The data was collected during task4. As described in task4, the images were downloaded in AWS S3 bucket and the labels are included in the images’s names whose format is as follows:<br/>
 *'camera-id'\_r'roadConditionCategory'\_w'weatherConditionCategory'\_'measuredTime'*<br/>
 eg. "C1255201_r7_w0_2020-01-29_21-00-39"<br/>
 The weather conditions to classify are:<br/>
 1. Clear (0)
 2. Raining, three scales:
      * Weak rain (1)
      * Mediocre rain (2)
      * Heavy rain (3)
  3. Snowing, three scales:
      * Weak snow/sleet (4)
      * Mediocre snow/sleet (5)
      * Heavy snow/sleet (6)
      
Unfortunately the labels are not accurate and have many mistakes and that’s due to different reasons such as the quality of the image, the distance between camera and weather station, sensors errors… so manually checking the labels was necessary. Besides, some categories (like mediocre rain) don’t exist in the collected dataset and some others have small amount of images. That’s why extra data from other cctv cameras was fed to the model. The sources of the added data could be found in ‘added_data.txt’ file.
# Training the model (train.py)
Once the data was ready, a model was built with tensorflow. I used the resnet50 architecture pretrained on imagenet dataset. The choice of the architecture was based on the fact that the model must be light weighted in order to be run in realtime on a Jetson Nano device. Therefore, I had to make a compromise between accuracy and lesser number of parameters. Since depth-wise convolutions are known of low accuracy, I didn’t opt for mobilenet. So I found that resnet50 (less layers is not available in Keras) is the best candidate.<br/>  
The data was augmented (horizontal_flip, rotation, width_shift, height_shift) using keras library.<br/>
For the model finetuning grid search was applied to these hyperparameters:
* **learning_rate**: The learning rate of Adam algorithm
* **batch_size**: Training batch size
* **image_size**: Chosen size of image
* **layers_fine_tune**: Layers to be unfrozen and fine tuned

The best validation accuracy I got is **0.97** with training accuracy of **0.97** (execution **#212** in Valohai).<br/>
This model was obtained with learning rate equal to *0.001*, batch size of *64* sample, image with *(224x224)* size and no layer fine tuned.
# Testing the model (predict.py)
To test the performance of the model we run the model on images not included in training and validation datasets.
## Prediction on images
You can predict on images using *predict_images_tf.py* script:
```sh
python3 predict_images_tf.py --predictions weather --model ./models/tensorflow/WeatherCondi.h5 --input ./input --output ./output_weather --labels ./weather_labels.json
```
Where you mention on **'prediction'** argument that you want to predict weather conditions and put the path of the training model on **'model'** argument. The **'input'** argument is the path of your input images and the **'output'** is the path of the output images. As for **'labels'** argument you put the path of *'weather_labels.json'* file

## Prediction on video
You can predict on video using *predict_video_tf.py* script:
```sh
python3 predict_video_tf.py --predictions weather --model ./models/tensorflow/WeatherCondi.h5 --input ./test_video.mp4 --labels ./weather_labels.json --output ./output/weather_conditions.avi --size 1
```
Where you mention on **'prediction'** argument that you want to predict weather conditions and put the path of the training model on **'model'** argument. The **'input'** argument is the path of your input video and the **'output'** is the path of the output video. As for **'labels'** argument you put the path of *'weather_labels.json'* file
