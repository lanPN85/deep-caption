from keras.preprocessing.image import load_img, img_to_array

import numpy as np
import os
import json
import cv2


def load_image(path, size=(128, 128), normalize=True):
    return img_to_array(load_img(path, target_size=size))


def load_image_vgg(path, size=(224, 224)):
    img = cv2.resize(cv2.imread(path), size)

    mean_pixel = [103.939, 116.779, 123.68]
    img = img.astype(np.float32, copy=False)
    for c in range(3):
        img[:, :, c] = img[:, :, c] - mean_pixel[c]

    return img


def load_image_resnet(path, size=(224, 224)):
    img = cv2.resize(cv2.imread(path), size)

    img = img.astype(np.float32, copy=False)
    img = img[:, :, [2, 1, 0]]

    return img


def get_image_paths(directory, extensions=('png', 'jpg')):
    files = os.listdir(directory)
    files = list(map(lambda x: os.path.join(directory, x), files))
    img_files = list(filter(lambda x: x.split('.')[-1] in extensions, files))
    return img_files


def get_image_ids(annotation_path, img_paths):
    with open(annotation_path, 'rt') as f:
        js = json.load(f)
        name2id = {}
        for image in js['images']:
            name2id[image['file_name']] = image['id']

        ids = []
        for img in img_paths:
            fname = os.path.split(img)[-1]
            ids.append(name2id[fname])

    return ids


def get_captions(captions_path, img_paths):
    path2cap = {}
    with open(captions_path, 'rt') as f:
        for line in f:
            cols = line.split('\t')
            if len(cols) < 2:
                continue
            path2cap[cols[0]] = cols[1].strip()

    captions = []
    for ip in img_paths:
        fname = os.path.split(ip)[1]
        captions.append(path2cap[fname])
    return captions


def char_tokenize(sentence):
    return list(sentence)


if __name__ == '__main__':
    a = load_image('../data/download.jpg')
