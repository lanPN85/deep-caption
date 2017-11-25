from PIL import Image

import numpy as np
import os
import json


def load_image(path, size=(128, 128)):
    image = Image.open(path)
    image = image.resize(size, Image.ANTIALIAS)
    image = image.convert('RGB')

    imarr = np.array(image)
    imarr = imarr.astype(np.float32)

    imarr /= 255.0
    return imarr


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
