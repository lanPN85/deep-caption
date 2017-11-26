import sys
sys.path.append('.')
import numpy as np
import datetime
import math

from caption.model import CaptionModel
from caption.topologies import VGG_TRUNC, LSTM1
from caption.vocab import Vocab
from caption import utils

vocab = Vocab(tokenizer=utils.char_tokenize)
train_img = utils.get_image_paths('./data/coco/train2014')
train_captions = utils.get_captions('./data/coco/train2014/captions.txt', train_img)
vocab.build(train_captions)

m = CaptionModel(VGG_TRUNC, LSTM1, vocab)

# m.build(1000)
# m.summary()

count = 0
t0 = datetime.datetime.now()
for i, o, ms in m._generate_batch(train_img, train_captions, batch_size=60):
    t1 = datetime.datetime.now()
    print(' %d [%d ms]' % (count, (t1 - t0).microseconds / 1e3))
    count += 1
    if count % math.ceil(len(train_captions) / 60) == 0:
        input('Epoch complete. ')
    t0 = datetime.datetime.now()

# inp = np.ones((10, 128, 128, 3))
# lbl = np.ones((10, 20, 1000))
# m.compile()
# m.model.fit(inp, lbl, batch_size=1, epochs=2)
