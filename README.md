# AM-Traffic-I-Phase-2-Iteration-2-Task1-Tensorflow

Prepare and develop end-to-end pipeline for a weather conditions classification light-weight neural network. 

As an input this model should take a video sequence from CCTV camera; As an output model should classify weather conditions (Clear, Rain, Snow). 

Network should be light enough to run in realtime on a Jetson Nano device.

-------------------------------------------------------------------------------------------------------------------------------

# Data
The data was collected during task4. As described in task4, the images were downloaded in AWS S3 bucket and the labels are included in the images’s names whose format is as follows:<br/>
 ‘camera-id’_r’roadConditionCategory’_w’weatherConditionCategory’_’measuredTime’/ eg. “C1255201_r7_w0_2020-01-29_21-00-39”<br/>
 The weather conditions to classify are:
 a. Clear (0)
 b. Raining, three scales:
  *Weak rain (1)
  *Mediocre rain (2)
  *Heavy rain (3)
  c. Snowing, three scales:
    *Weak snow/sleet (4)
    *Mediocre snow/sleet (5)
    *Heavy snow/sleet (6)
