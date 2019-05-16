import numpy as np
import os

from keras.preprocessing import image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

def extract_features(model, x):
    features = model.predict(x)
    return features


def extract_features_vgg16(imgdir):
    features_list = []
    model = VGG16(weights='imagenet', include_top=False)
    for imgfile in os.listdir(imgdir):
        img_path = os.path.join(imgdir, imgfile)
        # the images can be of any shape and don't have to be reduced to 224 x 224
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = vgg_preprocess_input(x)
        feat = extract_features(model, x)
        features_list.append(feat.flatten())
    return np.array(features_list)


def extract_features_resnet(imgdir):
    features_list = []
    model = ResNet50(weights='imagenet', include_top=False)
    for imgfile in os.listdir(imgdir):
        img_path = os.path.join(imgdir, imgfile)
        # the images can be of any shape and don't have to be reduced to 224 x 224
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = resnet_preprocess_input(x)
        feat = extract_features(model, x)
        features_list.append(feat.flatten())
    return np.array(features_list)


if __name__ == '__main__':
    # To extract features from VGG-16
    features = extract_features_vgg16('data/train')
    np.save('features_train.npy')
    features = extract_features_vgg16('data/val')
    np.save('features_val.npy')

    # UNCOMMENT to run feature extraction from ResNet 50
    features = extract_features_resnet('data/train')
    np.save('features_train.npy')
    features = extract_features_resnet('data/val')
    np.save('features_val.npy')