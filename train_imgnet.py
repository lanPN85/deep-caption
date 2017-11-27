from argparse import ArgumentParser
from keras.optimizers import Adam

import os
import nltk

from caption import CaptionModel, Vocab
from caption.topologies import *
from caption.topo import *
from caption import utils

DEFAULT_LR = 0.001
DEFAULT_DROPOUT = 0.0
DEFAULT_SENTENCE_LEN = 30
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
DEFAULT_VOCAB_MIN = 2

IMAGE_SIZE = (224, 224)
MAP = {
    'vgg': vgg_imgnet_decode,
    'resnet': resnet_imgnet_decode
}


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
    parser.add_argument('--vocab-min', default=DEFAULT_VOCAB_MIN, type=int, dest='VMIN')
    parser.add_argument('--mode', default='word', dest='MODE')
    parser.add_argument('--connector', default=1000, type=int, dest='CONN')
    parser.add_argument('--type', default='vgg', dest='TYPE')

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
    if args.MODE == 'word':
        tokenizer = nltk.word_tokenize
    elif args.MODE == 'char':
        tokenizer = utils.char_tokenize
    else:
        raise ValueError('Unknown mode %s.' % args.MODE)

    vocab = Vocab(tokenizer=tokenizer)
    vocab.build(train_docs, wmin=args.VMIN)

    print('Creating model...')
    model = CaptionModel(None, None, vocab, IMAGE_SIZE, sentence_len=args.SENTENCE_LEN,
                         dropout=args.DROPOUT, save_dir=args.MODEL_DIR, img_loader=utils.load_image_vgg,
                         connector_dim=args.CONN)
    model.model = MAP[args.TYPE](model)
    model.summary()

    print('Compiling...')
    model.compile(Adam(lr=args.LR))

    print('Starting training...')
    try:
        model.train(train_img, train_docs, val_img, val_docs, epochs=args.EPOCHS,
                    batch_size=args.BATCH, initial_epoch=args.FROM)
    except KeyboardInterrupt:
        print('\nTraining interrupted.')


if __name__ == '__main__':
    main(parse_arguments())
