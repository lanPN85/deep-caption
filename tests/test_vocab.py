import sys
sys.path.append('.')

from caption.vocab import Vocab
from caption import utils

v = Vocab()
v.build(['Hello world'])
