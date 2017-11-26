import os
import matplotlib.pyplot as plt

from argparse import ArgumentParser

from caption import utils, CaptionModel


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(dest='MODEL')
    parser.add_argument('--output', '-o', default=None, dest='OUT')
    parser.add_argument('--image-dir', '-d', default=None, dest='DATA')
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
            plt.ion()
            while True:
                cwd = os.getcwd()
                query = input('%s> ' % cwd)
                if not os.path.exists(query):
                    print(' Invalid path to file.')
                    continue
                caption = model.caption(query)
                print('-> %s' % caption)
                if args.SHOW:
                    mat = utils.load_image(query, normalize=False)
                    plt.imshow(mat)

        except (KeyboardInterrupt, EOFError):
            print()


if __name__ == '__main__':
    main(parse_arguments())
