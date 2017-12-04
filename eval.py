import os
import numpy as np

from PIL import Image
from argparse import ArgumentParser

from caption import utils, CaptionModel


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(dest='MODEL')
    parser.add_argument('--output', '-o', default=None, dest='OUT')
    parser.add_argument('--image-dir', '-d', default=None, dest='DATA')
    parser.add_argument('--annotation', default='./data/coco/captions_val2014.json', dest='ANN')
    parser.add_argument('--show', action='store_true', dest='SHOW')

    return parser.parse_args()


def main(args):
    model = CaptionModel.load(args.MODEL)
    model.summary()

    if not args.OUT and not args.DATA:
        print('Starting shell...')
        print('Enter a file path to display it and the predicted caption.')
        print('Press Ctrl+C or Ctrl+D to exit.')
        try:
            while True:
                cwd = os.getcwd()
                query = input('%s> ' % cwd)
                if not os.path.exists(query):
                    print(' Invalid path to file.')
                    continue
                caption = model.caption(query)
                print('-> %s' % caption)
                if args.SHOW:
                    with Image.open(query) as img:
                        img.show()
        except (KeyboardInterrupt, EOFError):
            print()
    elif args.DATA:
        print('Fetching metadata...')
        img_paths = utils.get_image_paths(args.DATA)
        img_ids = utils.get_image_ids(args.ANN, img_paths)

        if not args.OUT:
            print('Generating captions...')
            try:
                for img in img_paths:
                    cap = model.caption(img)
                    print('[%s]\n%s' % (img, cap))
                    if args.SHOW:
                        with Image.open(img) as image:
                            image.show()
                    input('Next image...')
            except (KeyboardInterrupt, EOFError):
                print()


if __name__ == '__main__':
    main(parse_arguments())
