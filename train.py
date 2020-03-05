import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Dense
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import os
from PIL import Image, ImageFile
#--------------------------------------Building the model--------------------------------------
num_classes = 7 # number of classes

model = Sequential()
resNet = ResNet50(include_top=False, pooling='avg', weights='imagenet') # pretrained model

# froze layers in resnet
layers_fine_tune = -1
for layer in resNet.layers: #[:layers_fine_tune]: 
	layer.trainable = False
model.add(resNet) 
model.add(Dense(num_classes, activation='softmax')) # Classification layer

# Compiling the model
adam = Adam(lr=0.005)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

#--------------------------------------Preprocessing data--------------------------------------
image_size = 512 # chosen size of images

# Data augmentation for only the training dataset 
data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   rotation_range=20,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2)

train_generator = data_generator_with_aug.flow_from_directory(
        '/valohai/inputs/dataset/dataset/train',
        target_size=(image_size, image_size),
        batch_size=64,
        class_mode='categorical')

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = data_generator_no_aug.flow_from_directory(
        '/valohai/inputs/dataset/dataset/val',
        target_size=(image_size, image_size),
        class_mode='categorical')
ImageFile.LOAD_TRUNCATED_IMAGES = True
#--------------------------------------Training and saving the model--------------------------------------
model_name='WeatherCondi-{val_acc:.2f}'

# we save the best model (with maximum validation accuracy)
checkpointer = ModelCheckpoint(
    filepath=os.path.join('/valohai/outputs', model_name + '.h5'),
    monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)


history=model.fit_generator(
        train_generator,
        steps_per_epoch=5,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=1,
        callbacks=[checkpointer])
