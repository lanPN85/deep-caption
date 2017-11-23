from PIL import Image

import numpy as np


def load_image(path, size=(128, 128)):
    image = Image.open(path)
    image = image.resize(size, Image.ANTIALIAS)

    imarr = np.array(image)
    imarr = imarr.astype(np.float32)
    imarr /= 255.0
    return imarr


if __name__ == '__main__':
    a = load_image('../data/download.jpg')
