import sys
sys.path.append('.')
import numpy as np

from caption.model import CaptionModel
from caption.topologies import VGG_TRUNC, LSTM1

m = CaptionModel(VGG_TRUNC, LSTM1)

m.build(1000)
m.summary()

inp = np.ones((10, 128, 128, 3))
lbl = np.ones((10, 20, 1000))
m.compile()
m.model.fit(inp, lbl, batch_size=1, epochs=2)
