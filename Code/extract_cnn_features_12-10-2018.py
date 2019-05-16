import numpy as np
import os
from keras import backend as K

from keras.preprocessing import image

from keras.models import Sequential, load_model, Model
#from keras.utils import load_img

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator


if K.image_data_format() == 'channels_first':
    input_shape = (3, 224, 224)
else:
    input_shape = (224, 224, 3)

def extract_features(model, x):
    features = model.predict(x)
    return features


def extract_features_vgg16_finetuned(weights_file, imgdir):

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

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Conv2D(4096, (3, 3), strides=3, padding="valid", activation='relu'))
    model.add(Conv2D(4096, (1, 1), strides=1, padding="valid", activation='relu'))
    model.add(Conv2D(3, (1, 1), strides=1, padding="valid", activation='relu'))
    model.add(AveragePooling2D((3, 3)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.load_weights(weights_file, by_name=True)

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('conv2d_15').output)

    datagen = ImageDataGenerator()

    generator = datagen.flow_from_directory(
        imgdir,
        target_size=(224, 224),
        batch_size=16,
        class_mode=None,  # only data, no labels
        shuffle=False)  # keep data in same order as labels

    features = intermediate_layer_model.predict_generator(generator)
    features = np.mean(features, (1,2))
    return features

if __name__ == '__main__':

    # To extract features from VGG-16
    features = extract_features_vgg16_finetuned('model/univef_VGG16.h5', 'data/val')
    np.save('features_vgg16_finetuned.npy')