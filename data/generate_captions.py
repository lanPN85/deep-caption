from argparse import ArgumentParser

import json


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--train-file', default='coco/captions_train2014.json', dest='TRAIN')
    parser.add_argument('--val-file', default='coco/captions_val2014.json', dest='VAL')
    parser.add_argument('--train-out', default='coco/train2014/captions.txt', dest='TRAIN_OUT')
    parser.add_argument('--val-out', default='coco/val2014/captions.txt', dest='VAL_OUT')

    return parser.parse_args()


def main(args):
    print('Creating train annotations...')
    with open(args.TRAIN) as f:
        js = json.load(f)
        annotations = js['annotations']
        images = js['images']
        id2name = {}
        print('Pairing ids...')
        for img in images:
            id2name[img['id']] = img['file_name']

        print('Writing file...')
        with open(args.TRAIN_OUT, 'wt') as fout:
            for ann in annotations:
                s = '%s\t%s\n' % (id2name[ann['image_id']], ann['caption'])
                fout.write(s)
                print(s, end='')

    print('Creating validation annotations...')
    with open(args.VAL) as f:
        js = json.load(f)
        annotations = js['annotations']
        images = js['images']
        id2name = {}
        print('Pairing ids...')
        for img in images:
            id2name[img['id']] = img['file_name']

        print('Writing file...')
        with open(args.VAL_OUT, 'wt') as fout:
            for ann in annotations:
                s = '%s\t%s\n' % (id2name[ann['image_id']], ann['caption'])
                fout.write(s)
                print(s, end='')


if __name__ == '__main__':
    main(parse_arguments())
