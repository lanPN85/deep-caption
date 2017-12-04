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


def get_image_and_captions(annotations_path, img_paths, root_dir):
    id2path = {}
    new_paths = []
    captions = []
    with open(annotations_path, 'rt') as f:
        js = json.load(f)
        for im in js['images']:
            _path = os.path.join(root_dir, im['file_name'])
            if _path in img_paths:
                id2path[im['id']] = _path

        for cap in js['annotations']:
            if cap['image_id'] in id2path.keys():
                captions.append(cap['caption'])
                new_paths.append(id2path[cap['image_id']])

    return captions, new_paths


def char_tokenize(sentence):
    return list(sentence)


if __name__ == '__main__':
    a = load_image('../data/download.jpg')
