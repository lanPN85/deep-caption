import nltk
import numpy as np

NULL_TOKEN = ''


class Vocab:
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else []
        self._w2i = {NULL_TOKEN: 0}
        self._i2w = [NULL_TOKEN]

    def build(self, reset=True):
        if reset:
            self._w2i = {NULL_TOKEN: 0}
            self._i2w = [NULL_TOKEN]

        for d in self.docs:
            for w in nltk.word_tokenize(d):
                if w not in self._w2i.keys():
                    self._w2i[w] = len(self._i2w)
                    self._i2w.append(w)

    def encode_sentence(self, sentence, length=20):
        mat = np.zeros((length, self.size))
        tokens = nltk.word_tokenize(sentence)
        tokens += [NULL_TOKEN] * (length - len(tokens))
        for i, w in enumerate(tokens):
            mat[i, self[w]] = 1
        return mat

    @staticmethod
    def mask_sentence(sentence, length=20):
        mask = np.ones((length, 1))
        tokens = nltk.word_tokenize(sentence)
        for i in range(len(tokens), length):
            mask[i] = 0.0
        return mask

    def __getitem__(self, item):
        if type(item) == int:
            return self._i2w[item]
        else:
            return self._w2i.get(item, self._w2i[NULL_TOKEN])

    @property
    def size(self):
        return len(self._i2w)
