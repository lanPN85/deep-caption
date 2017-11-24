from argparse import ArgumentParser
from keras.optimizers import RMSprop

import os

from caption import CaptionModel, Vocab
from caption.topologies import *
from caption import utils


DEFAULT_LR = 0.001
DEFAULT_DROPOUT = 0.0
DEFAULT_SENTENCE_LEN = 30
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
DEFAULT_VOCAB_LIMIT = 5000

IMAGE_SIZE = (128, 128)

CONV_TOPO = VGG
LSTM_TOPO = LSTM2


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('--lr', default=DEFAULT_LR, type=float, dest='LR')
    parser.add_argument('--dropout', default=DEFAULT_DROPOUT, type=float, dest='DROPOUT')
    parser.add_argument('--train-dir', default='./data/coco/train2014', dest='TRAIN_DIR')
    parser.add_argument('--val-dir', default='./data/coco/val2014', dest='VAL_DIR')
    parser.add_argument('--model-dir', '-md', default='./models/default', dest='MODEL_DIR')
    parser.add_argument('--sent-len', '-sl', default=DEFAULT_SENTENCE_LEN, type=int, dest='SENTENCE_LEN')
    parser.add_argument('--epochs', '-e', default=DEFAULT_EPOCHS, type=int, dest='EPOCHS')
    parser.add_argument('--from', default=0, type=int, dest='FROM')
    parser.add_argument('--batch-size', default=DEFAULT_BATCH, type=int, dest='BATCH')
    parser.add_argument('--cutoff', default=None, type=int, dest='CUTOFF')
    parser.add_argument('--vocab', default=DEFAULT_VOCAB_LIMIT, type=int, dest='VLIMIT')

    return parser.parse_args()


def main(args):
    print('Fetching image paths...')
    train_img = utils.get_image_paths(args.TRAIN_DIR)
    val_img = utils.get_image_paths(args.VAL_DIR)
    if args.CUTOFF is not None:
        train_img = train_img[:args.CUTOFF]
        val_img = val_img[:args.CUTOFF]

    print('Reading captions...')
    train_docs = utils.get_captions(os.path.join(args.TRAIN_DIR, 'captions.txt'), train_img)
    val_docs = utils.get_captions(os.path.join(args.VAL_DIR, 'captions.txt'), val_img)

    print('Building vocabulary...')
    vocab = Vocab()
    vocab.build(train_docs, limit=args.VLIMIT)
    print(' Vocabulary size: %d' % vocab.size)

    print('Creating model...')
    model = CaptionModel(CONV_TOPO, LSTM_TOPO, vocab, img_size=IMAGE_SIZE,
                         sentence_len=args.SENTENCE_LEN, dropout=args.DROPOUT,
                         save_dir=args.MODEL_DIR)
    model.build()
    model.summary()

    print('Compiling...')
    model.compile(RMSprop(lr=args.LR))

    print('Starting training...')
    try:
        model.train(train_img, train_docs, val_img, val_docs, epochs=args.EPOCHS,
                    batch_size=args.BATCH, initial_epoch=args.FROM)
    except KeyboardInterrupt:
        print('\nTraining interrupted.')


if __name__ == '__main__':
    main(parse_arguments())
