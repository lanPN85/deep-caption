import sys
sys.path.append('.')

from caption.vocab import Vocab
from caption import utils

v = Vocab()
train_img = utils.get_image_paths('./data/coco/train2014/')
val_img = utils.get_image_paths('./data/coco/val2014/')

train_docs = utils.get_captions('./data/coco/train2014/captions.txt', train_img)
val_docs = utils.get_captions('./data/coco/val2014/captions.txt', val_img)
v.build(train_docs + val_docs, wmin=5)
