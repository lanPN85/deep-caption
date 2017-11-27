import nltk
import numpy as np


class Vocab:
    NULL_TOKEN = ''
    END_TOKEN = '\n'

    def __init__(self, tokenizer=nltk.word_tokenize):
        self.tokenizer = tokenizer
        self._w2i = {self.NULL_TOKEN: 0, self.END_TOKEN: 1}
        self._i2w = [self.NULL_TOKEN, self.END_TOKEN]

    def build(self, docs, reset=True, wmin=5):
        if reset:
            self._w2i = {self.NULL_TOKEN: 0, self.END_TOKEN: 1}
            self._i2w = [self.NULL_TOKEN, self.END_TOKEN]

        freq = {}
        for d in docs:
            for w in self.tokenizer(d.lower()):
                if w not in freq.keys():
                    freq[w] = 1
                else:
                    freq[w] = freq[w] + 1

        for w, f in freq.items():
            if f >= wmin:
                assert w not in self._w2i.keys()
                self._w2i[w] = len(self._i2w)
                self._i2w.append(w)

        print(' Documents scanned: %d' % len(docs))
        print(' Total possible words: %d' % len(freq.keys()))
        print(' Vocabulary size: %d' % self.size)

    def encode_sentence(self, sentence, length=20):
        mat = np.zeros((length, self.size))
        if type(sentence) == str:
            tokens = self.tokenizer(sentence.lower())[:length-1]
        else:
            tokens = sentence[:length-1]
        tokens += [self.END_TOKEN] + [self.NULL_TOKEN] * (length - 1 - len(tokens))
        for i, w in enumerate(tokens):
            mat[i, self[w]] = 1.0
        return mat

    def encode_word(self, word):
        vec = np.zeros((self.size,))
        vec[self[word]] = 1.0
        return vec

    def mask_sentence(self, sentence, length=20):
        mask = np.ones((length,))
        if type(sentence) == str:
            tokens = self.tokenizer(sentence.lower())[:length-1]
        else:
            tokens = sentence[:length-1]
        tokens += [self.END_TOKEN]
        for i in range(len(tokens), length):
            mask[i] = 0.0
        return mask

    def __getitem__(self, item):
        if type(item) in (int, np.int64, np.int32):
            return self._i2w[item]
        else:
            return self._w2i.get(item, self._w2i[self.NULL_TOKEN])

    @property
    def size(self):
        return len(self._i2w)
