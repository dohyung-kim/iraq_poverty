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

from sklearn.preprocessing import LabelBinarizer
from sklearn import mixture

import os, json
import cv2
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

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
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
    #model.load_weights('C:/Users/Adyasha/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    model.load_weights('model/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
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

    model.add(ZeroPadding2D((1, 1)))
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
    adam = Adam(lr = 1e-4, decay=1e-6, momentum=0.9)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def poverty_cnn():

    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(24, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('sigmoid'))

    # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr = 1e-4, decay=1e-6, momentum=0.9)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

def load_train_data(image_dir, color_mode='rgb', val_ratio=0.2):
    pass

def load_val_data(image_dir, color_mode='rgb'):
    pass

def load_ntl(ntlpath):

    from PIL import Image
    im = Image.open(ntlpath)
    imarray = np.array(im)
    # check
    # print(imarray.shape)
    # print(imarray[:10])

    tfwfile = ntlpath.replace('tif', 'tfw')
    with open(tfwfile, 'r') as f:
        lines = f.readlines()
    lines = [float(line.strip()) for line in lines]
    col_inc, _, _, row_inc, lon_init, lat_init = lines

    latbyrow = np.zeros(imarray.shape[0], dtype=np.float32)
    lonbycol = np.zeros(imarray.shape[1], dtype=np.float32)

    for r in range(0, imarray.shape[0]):
        latbyrow[r] = row_inc*r + lat_init
    for c in range(0, imarray.shape[1]):
        lonbycol[c] = col_inc*c + lon_init

    # check
    # print(latbyrow[:10])
    # print(lonbycol[:10])

    return imarray, latbyrow, lonbycol

def load_survey_data(filepath):
    df = pd.read_csv(filepath)



def run_gmm(samples):
    X_raw = np.array(samples)
    m = np.mean(X_raw)
    s = np.std(X_raw)
    print(m, s)
    X = (X_raw - m) / s
    print(X.shape)
    samples = X.reshape(-1, 1)

    lowest_bic = np.infty
    bic = []
    n_components = 3
    cv_type = 'spherical'
    for i in range(0, 5):
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(samples)
        bic.append(gmm.bic(samples))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
        print(bic)

    clf = best_gmm
    print(clf.means_, clf.covariances_, clf.weights_)

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    sum_gauss = np.linspace(0, 0, 1000)
    for mu, sig, w in zip(clf.means_, np.transpose(clf.covariances_), np.transpose(clf.weights_)):
        gcomp = w * gaussian(np.linspace(-6, 6, 1000), mu, sig ** (0.5))
        sum_gauss = sum_gauss + gcomp
        plt.plot(np.linspace(-6, 6, 1000), gcomp, linewidth=1)

    plt.plot(np.linspace(-6, 6, 1000), sum_gauss, linewidth=1)
    plt.ylabel('Probability')
    plt.xlabel('Nightlights')
    plt.savefig('gmm-nightlights.png')

    return clf, m, s

if __name__ == '__main__':

    # Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 400, 400 # Resolution of inputs
    channel = 3
    batch_size = 64
    nb_epoch = 25

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    image_dir = 'data'
    # X_train, y_train, X_val, y_val, val_filenames = load_train_data(image_dir)
    # X_train = preprocess_input(X_train.astype(float))
    # X_val = preprocess_input(X_val.astype(float))
    # lb = LabelBinarizer()
    # y_train = lb.fit_transform(y_train)
    # y_val = lb.transform(y_val)

    ntlarray, lats, lons = load_ntl("data/NTL/NTL/7d1b3ba62de8cf46d5ea2b6cef766b2a.avg_rad.tif")
    samples = ntlarray.flatten()
    #samples = np.expand_dims(samples, axis=1)
    _ = run_gmm(samples)
    sys.exit()

    # Load our model
    model = vgg16_modified(224, 224)
    # model = poverty_cnn(20, 20)

    now = datetime.now()
    tensorboard_cb = TensorBoard(log_dir=os.path.join('logs', now.strftime("%Y%m%d-%H%M%S")), histogram_freq=0,
                                 write_grads=False, write_graph=False, write_images=True, batch_size=32)

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    checkpoint = ModelCheckpoint(filepath='predict_poverty_checkpoint.h5',
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')
    train_data_generator = datagen.flow_from_directory(
        'data/train',
        target_size=(224,224),
        batch_size=32,
        class_mode='binary'
    )
    val_data_generator = datagen.flow_from_directory(
        'data/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    # X_val, y_val, val_filenames = load_val_data('data/val')
    
    nb_train_samples = _
    nb_val_samples = _
    # Start Fine-tuning
    model.fit_generator(
        train_data_generator,
        nb_epoch=nb_epoch,
        steps_per_epoch=nb_train_samples // 32,
        #steps_per_epoch=3,
        verbose=1,
        # validation_data=(X_val, y_val),
        validation_data=val_data_generator,
        validation_steps=nb_val_samples // 32,
        #validation_steps=2,
        callbacks=[tensorboard_cb, checkpoint]
    )

    # Make predictions
    # val_predictions = model.predict(X_val, batch_size=batch_size, verbose=1)
    # val_predictions = val_predictions.tolist()
    # pred_tuples = list(zip(val_filenames, val_predictions))
    # with open('val-predictions.json', 'w') as f:
    #     json.dump(pred_tuples, f)

    # Cross-entropy loss score
    # score = log_loss(y_val, val_predictions)

    model.save_weights('predict_poverty_VGG16.h5')