import sys
sys.path.append('.')

from caption.vocab import Vocab
from caption import utils

v = Vocab(tokenizer=utils.char_tokenize)
v.build(['abcde', '0123'])
