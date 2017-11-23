from PIL import Image

import numpy as np
import os


def load_image(path, size=(128, 128)):
    image = Image.open(path)
    image = image.resize(size, Image.ANTIALIAS)

    imarr = np.array(image)
    imarr = imarr.astype(np.float32)
    imarr /= 255.0
    return imarr


def get_image_paths(directory, extensions=('png', 'jpg')):
    files = os.listdir(directory)
    img_files = list(filter(lambda x: x.split('.')[-1] in extensions, files))
    return img_files


def get_captions(filepath):
    with open(filepath, 'rt') as f:
        lines = f.readlines()


if __name__ == '__main__':
    a = load_image('../data/download.jpg')
