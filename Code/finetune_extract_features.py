# -*- coding: utf-8 -*-

from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
#from keras.utils import load_img

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend as K
from datetime import datetime
import random
from shutil import copy
import matplotlib.pyplot as plt

from sklearn import mixture
from sklearn.preprocessing import LabelBinarizer


import os, json
#import cv2
import numpy as np
import pandas as pd

import sys
import os

def vgg16_modified(img_rows, img_cols, channel=3):
    """VGG 16 Model for Keras

    Model Schema is based on
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of categories for our classification task
    """
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 224, 224)
    else:
        input_shape = (224, 224, 3)
        
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    #model.load_weights('model/vgg16_weights.h5')
    model.load_weights('/home/ubuntu/model/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    # Truncate and replace softmax layer for transfer learning

    print("Original VGG-16")
    print(model.summary())

    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()

    print("VGG-16 after the fully convolutional layers have been removed")
    print(model.summary())

    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Conv2D(4096, (3, 3), strides=3, padding="valid", activation='relu'))
    model.add(Conv2D(4096, (1, 1), strides=1, padding="valid", activation='relu'))
    model.add(Conv2D(3, (1, 1), strides=1, padding="valid", activation='relu'))
    model.add(AveragePooling2D((3, 3)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    print("VGG-16 after being converted to a fully convolutional model")
    print(model.summary())

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr = 1e-4, decay=1e-6) #, momentum=0.9
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_train_data(image_dir, color_mode='rgb', val_ratio=0.2):
    pass

def load_val_data(image_dir, color_mode='rgb'):
    pass



if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    batch_size = 64
    nb_epoch = 25

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    image_dir = '/home/ubuntu/data/train/'
    #X_train, y_train, X_val, y_val, val_filenames = load_train_data(image_dir)
    #print("I got here")
    #X_train = preprocess_input(X_train.astype(float))
    #X_val = preprocess_input(X_val.astype(float))
    #lb = LabelBinarizer()
    #y_train = lb.fit_transform(y_train)
    #y_val = lb.transform(y_val)
    
    

   # ntlarray, lats, lons = load_ntl("/Users/en22/Dropbox/UNICEF/Analyses/data/Nightlights/2012/convert_test.tif")
   # samples = ntlarray.flatten()
    #samples = np.expand_dims(samples, axis=1) #this was originally commented out
   # _ =run_gmm(samples)
   # sys.exit()

    # Load our model
    model = vgg16_modified(224, 224, 3)
    #model = poverty_cnn(20, 20)

    now = datetime.now()
    tensorboard_cb = TensorBoard(log_dir=os.path.join('logs', now.strftime("%Y%m%d-%H%M%S")), histogram_freq=0,
                                 write_grads=False, write_graph=False, write_images=True, batch_size=32)

    
    datagen = ImageDataGenerator(
       rotation_range=40,
       width_shift_range=0.2,
       height_shift_range=0.2,
       rescale=1./255,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True,
       fill_mode='nearest')
    
    checkpoint = ModelCheckpoint(filepath='predict_poverty_checkpoint.h5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    train_data_generator = datagen.flow_from_directory(
        '/home/ubuntu/data/train/high',
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='binary'
    )
    val_data_generator = datagen.flow_from_directory(
        '/home/ubuntu/data/val/high',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    #X_val, y_val, val_filenames = load_val_data('data/val')
    
    nb_train_samples = 888
    nb_val_samples = 222
    # Start Fine-tuning
    model.fit_generator(
        train_data_generator,
        epochs=nb_epoch, #nb_epoch=nb_epoch,
        steps_per_epoch=nb_train_samples // batch_size,
        #steps_per_epoch=3,
        verbose=1,
        # validation_data=(X_val, y_val),
        validation_data=val_data_generator,
        validation_steps=nb_val_samples // batch_size,
        #validation_steps=2,
                        #callbacks=[tensorboard_cb, checkpoint]
    )
    print("I finished fine-tunning")

    # Make predictions
    #val_predictions = model.predict(X_val, batch_size=batch_size, verbose=1)
    #val_predictions = val_predictions.tolist()
    #pred_tuples = list(zip(val_filenames, val_predictions))
    #with open('val-predictions.json', 'w') as f:
    #  json.dump(pred_tuples, f)

    # Cross-entropy loss score
    #score = log_loss(y_val, val_predictions)

    model.save_weights('/home/ubuntu/model/predict_poverty_VGG16.h5')
