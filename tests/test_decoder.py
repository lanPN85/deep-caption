import sys
sys.path.append('.')
import numpy as np

from keras.layers import Input
from keras.models import Model

from caption.layers import DecoderLSTM
from caption import CaptionModel, Vocab
from caption import topo

v = Vocab()
v.build(['Hello World'])
cm = CaptionModel(None, None, v, img_size=(224, 224),
                  sentence_len=20, save_dir='models/test')
cm.model = topo.vgg_imgnet_decode(cm)
cm.compile()
cm.summary()
inp = np.ones((10, 224, 224, 3))
lbl = np.ones((10, 20, 4))
cm.model.fit(inp, lbl, batch_size=2, epochs=2)
