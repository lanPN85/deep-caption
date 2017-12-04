from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard

import math
import numpy as np
import os
import json
import pickle
import keras.models

from . import utils
from .callbacks import CaptionCallback
from .vocab import Vocab
from .layers import *


class CaptionModel:
    def __init__(self, vocab, img_size=(128, 128),
                 sentence_len=20, dropout=0.0, connector_dim=1000, save_dir='models/default',
                 img_loader=utils.load_image):
        self.img_size = img_size
        self.sentence_len = sentence_len
        self.save_dir = save_dir
        self.dropout = dropout
        self.vocab = vocab
        self.connector_dim = connector_dim
        self.img_loader = img_loader

        self.model = Sequential()

    def compile(self, optimizer=RMSprop()):
        self.model.compile(optimizer, loss='categorical_crossentropy',
                           metrics=['acc'], sample_weight_mode='temporal')

    def train(self, img_paths, captions, val_img, val_captions,
              epochs=100, batch_size=10, initial_epoch=0):
        os.makedirs(self.save_dir, exist_ok=True)

        per_epoch = math.ceil(len(captions) / batch_size)
        val_per_epoch = math.ceil(len(val_captions) / batch_size)
        callbacks = [CaptionCallback(self, monitor='loss', samples=img_paths[:5] + val_img[:5]),
                     CSVLogger(os.path.join(self.save_dir, 'epochs.csv'))]

        _gen = self._generate_batch(img_paths, captions, batch_size)
        _val_gen = self._generate_batch(val_img, val_captions, batch_size)
        self.model.fit_generator(_gen, per_epoch, shuffle=False, max_queue_size=5,
                                 epochs=epochs, initial_epoch=initial_epoch,
                                 validation_data=_val_gen, validation_steps=val_per_epoch,
                                 callbacks=callbacks)

    def evaluate(self, test_img, test_captions, batch_size=10):
        steps = math.ceil(len(test_captions) / batch_size)
        metrics = self.model.evaluate_generator(self._generate_batch(test_img, test_captions, batch_size),
                                                steps=steps, max_queue_size=3)
        return metrics

    def caption(self, image):
        if type(image) == str:
            return self.caption(self.img_loader(image, self.img_size))
        else:
            assert image.shape[0] == self.img_size[0]
            assert image.shape[1] == self.img_size[1]

            m = np.zeros((1, self.img_size[0], self.img_size[1], 3))
            m[0, :, :, :] = image
            probs = self.model.predict(m, batch_size=1, verbose=0)
            word_idx = np.argmax(probs, axis=-1)
            caption = ''
            # noinspection PyTypeChecker
            for idx in word_idx[0]:
                word = self.vocab[idx]
                if word == Vocab.END_TOKEN:
                    break
                caption += word + ' '
            return caption

    def caption_batch(self, images, image_ids=None, to_json=False,
                      json_file=None, batch_size=32, verbose=1):
        steps = math.ceil(len(images) / batch_size)
        probs = self.model.predict_generator(self._generate_image_batch(images, batch_size=batch_size),
                                             steps, verbose=verbose, max_queue_size=2)
        word_idx = np.argmax(probs, axis=-1)
        captions = []
        for i in range(len(images)):
            _caption = ''
            for idx in word_idx[i, :]:
                word = self.vocab[idx]
                if word == Vocab.END_TOKEN:
                    break
                _caption += word + ' '
            captions.append(_caption)

        if not to_json:
            return captions
        else:
            assert len(captions) == len(image_ids)
            jd = []
            for _caption, _id in zip(captions, image_ids):
                jd.append({'image_id': _id, 'caption': _caption})
            if json_file is not None:
                with open(json_file, 'wt') as f:
                    json.dump(jd, f)

            return json.dumps(jd)

    def _generate_batch(self, img_paths, captions, batch_size):
        _img_mat = np.zeros((batch_size, self.img_size[0], self.img_size[1], 3))
        _cap_mat = np.zeros((batch_size, self.sentence_len, self.vocab.size))
        _cap_mask = np.zeros((batch_size, self.sentence_len))
        tokens = []
        for c in captions:
            tokens.append(self.vocab.tokenizer(c))

        while True:
            for i in range(batch_size, len(captions), batch_size):
                _img_paths = img_paths[i - batch_size:i]
                _captions = tokens[i - batch_size:i]

                for j, _path in enumerate(_img_paths):
                    _img_mat[j, :, :] = self.img_loader(_path, self.img_size)

                for j, _caption in enumerate(_captions):
                    _cap_mat[j, :, :] = self.vocab.encode_sentence(_caption, length=self.sentence_len)
                    _cap_mask[j, :] = self.vocab.mask_sentence(_caption, length=self.sentence_len)

                yield _img_mat, _cap_mat, _cap_mask

    def _generate_image_batch(self, img_paths, batch_size):
        _img_mat = np.zeros((batch_size, self.img_size[0], self.img_size[1], 3))
        while True:
            for i in range(batch_size, len(img_paths), batch_size):
                _img_paths = img_paths[i - batch_size:i]
                for j, _path in enumerate(_img_paths):
                    _img_mat[j, :, :] = self.img_loader(_path, self.img_size)

            yield _img_mat

    def summary(self):
        self.model.summary()

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        f1 = open(os.path.join(self.save_dir, 'configs.pkl'), 'wb')
        f2 = open(os.path.join(self.save_dir, 'vocab.pkl'), 'wb')

        self.model.save(os.path.join(self.save_dir, 'model.hdf5'))
        configs = {
            'img_size': self.img_size,
            'sentence_len': self.sentence_len,
            'dropout': self.dropout,
            'img_loader': self.img_loader,
            'save_dir': self.save_dir
        }
        pickle.dump(configs, f1)
        pickle.dump(self.vocab, f2)
        f1.close()
        f2.close()

    @classmethod
    def load(cls, load_dir):
        f1 = open(os.path.join(load_dir, 'configs.pkl'), 'rb')
        f2 = open(os.path.join(load_dir, 'vocab.pkl'), 'rb')

        configs = pickle.load(f1)
        vocab = pickle.load(f2)
        model = keras.models.load_model(os.path.join(load_dir, 'model.hdf5'),
                                        custom_objects={
                                            'DecoderLSTM': DecoderLSTM,
                                            'DecoderLSTMCell': DecoderLSTMCell
                                        })
        f1.close()
        f2.close()

        cm = cls(vocab, **configs)
        cm.model = model
        return cm

    @classmethod
    def migrate(cls, model_path, vocab_path, **configs):
        f = open(vocab_path, 'rb')
        vocab = pickle.load(f)
        f.close()

        model = keras.models.load_model(
            model_path,
            custom_objects={
                'DecoderLSTM': DecoderLSTM,
                'DecoderLSTMCell': DecoderLSTMCell
            }
        )

        cm = cls(vocab, **configs)
        cm.model = model
        return cm
