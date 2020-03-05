import os
import json
import argparse
from PIL import Image, ImageFile
from keras.callbacks import LambdaCallback
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Dense
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
#--------------------------------------Building the model--------------------------------------
#To stop showing the Tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='Chosen size of image',
    )
    parser.add_argument(
        '--layers_fine_tune',
        type=int,
        default=0,
        help='Layers to be unfrozen',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Training batch size (larger batches are usually more efficient on GPUs)',
    )
    flags = parser.parse_args()
    return flags

flags = parse_args()

num_classes = 7 # number of classes
epochs = 1

model = Sequential()
resNet = ResNet50(include_top=False, pooling='avg', weights='imagenet') # pretrained model

# froze layers in resnet
if flags.layers_fine_tune != 0:
    layers_fine_tune = -flags.layers_fine_tune
    for layer in resNet.layers[:layers_fine_tune]: 
        layer.trainable = False
else:
    for layer in resNet.layers:
        layer.trainable = False
model.add(resNet) 
model.add(Dense(num_classes, activation='softmax')) # Classification layer

# Compiling the model
adam = Adam(lr=flags.learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

#--------------------------------------Preprocessing data--------------------------------------
image_size = flags.image_size # chosen size of images

# Data augmentation for only the training dataset 
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   rotation_range=20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

train_generator = data_generator_with_aug.flow_from_directory(
        '/valohai/inputs/dataset/dataset/train',
        target_size=(image_size, image_size),
        batch_size=flags.batch_size,
        class_mode='categorical')

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = data_generator_no_aug.flow_from_directory(
        '/valohai/inputs/dataset/dataset/val',
        target_size=(image_size, image_size),
        class_mode='categorical')
ImageFile.LOAD_TRUNCATED_IMAGES = True
#--------------------------------------Training and saving the model--------------------------------------
model_name='WeatherCondi'

# we save the best model (with maximum validation accuracy)
checkpointer = ModelCheckpoint(
    filepath=os.path.join('/valohai/outputs', model_name + '.h5'),
    monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

json_logging_callback = LambdaCallback(
            on_epoch_begin=lambda epoch, logs: print('Epoch %s/%s' % ((int(epoch) + 1), epochs)),
            on_epoch_end=lambda epoch, logs: print(json.dumps({
                'epoch': int(epoch) + 1,
                'loss': str(logs['loss']),
                'accuracy': str(logs['acc']),
                'val_loss': str(logs['val_loss']),
                'val_accuracy': str(logs['val_acc']),
            })),
        )


model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=1,
    callbacks=[checkpointer, json_logging_callback])
