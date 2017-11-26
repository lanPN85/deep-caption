from keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger
from keras.preprocessing.image import load_img, img_to_array
from recurrentshop import RecurrentSequential, LSTMCell
from seq2seq import LSTMDecoderCell

import math
import numpy as np
import os
import json
import pickle
import keras.models

from . import utils
from .callbacks import CaptionCallback
from .vocab import Vocab


class CaptionModel:
    def __init__(self, conv_layers, lstm_layers, vocab, img_size=(128, 128),
                 sentence_len=20, dropout=0.0, save_dir='models/default'):
        self.img_size = img_size
        self.conv_layers = conv_layers
        self.lstm_layers = lstm_layers
        self.sentence_len = sentence_len
        self.save_dir = save_dir
        self.dropout = dropout
        self.vocab = vocab

        self.model = Sequential()

    def build(self, readout=False):
        self.model = Sequential()

        out_row, out_col = self.img_size
        conv_depth = None
        for i, cl in enumerate(self.conv_layers):
            if i == 0:
                self.model.add(Conv2D(
                    cl['filters'], cl['kernel'], strides=cl['strides'],
                    activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3),
                    padding='same'
                ))
            else:
                self.model.add(Conv2D(
                    cl['filters'], cl['kernel'], strides=cl['strides'],
                    activation='relu', padding='same'
                ))
            if 'pool' in cl.keys():
                self.model.add(MaxPooling2D(cl['pool']))
                out_col /= cl['pool']
                out_row /= cl['pool']
            if 'dense' in cl.keys():
                self.model.add(Dense(cl['dense'], activation='hard_sigmoid'))
                conv_depth = cl['dense']
                break

        conv_depth = self.conv_layers[-1]['filters'] if conv_depth is None else conv_depth
        self.model.add(Reshape((int(out_row * out_col), conv_depth)))

        rnn = RecurrentSequential(decode=True, output_length=self.sentence_len,
                                  readout=readout, implementation=2)

        for i, ll in enumerate(self.lstm_layers[:-1]):
            self.model.add(LSTM(ll['units'], activation='tanh', return_sequences=True,
                                recurrent_activation='hard_sigmoid', dropout=self.dropout,
                                recurrent_dropout=self.dropout, implementation=2, unroll=True))
            # rnn.add(Dropout(self.dropout))
            # rnn.add(LSTMCell(ll['units'], activation='tanh'))

        self.model.add(LSTM(self.lstm_layers[-1]['units'], activation='tanh', return_sequences=False,
                            recurrent_activation='hard_sigmoid', dropout=self.dropout,
                            recurrent_dropout=self.dropout, implementation=2, unroll=True))
        # rnn.add(Dropout(self.dropout))
        # rnn.add(LSTMCell(self.lstm_layers[-1]['units'], activation='tanh'))

        if readout:
            assert self.lstm_layers[-1]['units'] == self.vocab.size

        rnn.add(LSTMCell(self.vocab.size, activation='softmax'))
        # rnn.add(LSTM(self.vocab.size, activation='softmax', return_sequences=True))
        # rnn.add(LSTMDecoderCell(units=self.vocab.size, hidden_dim=self.vocab.size, activation='softmax'))
        self.model.add(rnn)
        # self.model = self.seq_model

    def compile(self, optimizer=RMSprop()):
        self.model.compile(optimizer, loss='categorical_crossentropy',
                           metrics=None, sample_weight_mode='temporal')

    def train(self, img_paths, captions, val_img, val_captions,
              epochs=100, batch_size=10, initial_epoch=0):
        os.makedirs(self.save_dir, exist_ok=True)

        per_epoch = math.ceil(len(captions) / batch_size)
        val_per_epoch = math.ceil(len(val_captions) / batch_size)
        callbacks = [CaptionCallback(self, monitor='loss', samples=img_paths[:5] + val_img[:5]),
                     EarlyStopping(monitor='loss', patience=5, verbose=1),
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
            return self.caption(utils.load_image(image, self.img_size))
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
        _im = np.zeros((len(images), self.img_size[0], self.img_size[1], 3))
        for i, img in enumerate(images):
            if type(img) == str:
                _im[i, :, :, :] = utils.load_image(img, self.img_size)
            else:
                _im[i, :, :, :] = img

        m = np.zeros((len(images), self.img_size[0], self.img_size[1], 3))
        probs = self.model.predict(_im, batch_size=batch_size, verbose=verbose)
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
                    # _img_mat[j, :, :] = utils.load_image(_path, size=self.img_size)
                    _img_mat[j, :, :] = img_to_array(load_img(_path, target_size=self.img_size))

                for j, _caption in enumerate(_captions):
                    _cap_mat[j, :, :] = self.vocab.encode_sentence(_caption, length=self.sentence_len)
                    _cap_mask[j, :] = self.vocab.mask_sentence(_caption, length=self.sentence_len)

                yield _img_mat, _cap_mat, _cap_mask

    def summary(self):
        self.model.summary()

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        f1 = open(os.path.join(self.save_dir, 'configs.pkl'), 'wb')
        f2 = open(os.path.join(self.save_dir, 'vocab.pkl'), 'wb')

        self.model.save(os.path.join(self.save_dir, 'model.hdf5'))
        configs = (self.img_size, self.dropout, self.sentence_len, self.save_dir, self.conv_layers, self.lstm_layers)
        pickle.dump(configs, f1)
        pickle.dump(self.vocab, f2)
        f1.close()
        f2.close()

    @classmethod
    def load(cls, load_dir):
        f1 = open(os.path.join(load_dir, 'configs.pkl'), 'rb')
        f2 = open(os.path.join(load_dir, 'vocab.pkl'), 'rb')

        img_size, dropout, sentence_len, save_dir, conv_layers, lstm_layers = pickle.load(f1)
        vocab = pickle.load(f2)
        model = keras.models.load_model(os.path.join(load_dir, 'model.hdf5'),
                                        custom_objects={
                                            'RecurrentSequential': RecurrentSequential,
                                            'LSTMDecoderCell': LSTMDecoderCell
                                        })
        f1.close()
        f2.close()

        cm = cls(conv_layers, lstm_layers, vocab, img_size=(128, 128),
                 sentence_len=sentence_len, dropout=dropout, save_dir=save_dir)
        cm.model = model
        return cm


class DirectCaptionModel(CaptionModel):
    def build(self, readout=False):
        self.model = Sequential()

        out_row, out_col = self.img_size
        total_size = -1
        for i, cl in enumerate(self.conv_layers):
            if i == 0:
                self.model.add(Conv2D(
                    cl['filters'], cl['kernel'], strides=cl['strides'],
                    activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3),
                    padding='same'
                ))
            else:
                self.model.add(Conv2D(
                    cl['filters'], cl['kernel'], strides=cl['strides'],
                    activation='relu', padding='same'
                ))
            if 'pool' in cl.keys():
                self.model.add(MaxPooling2D(cl['pool']))
                out_col /= cl['pool']
                out_row /= cl['pool']
            if 'dense' in cl.keys():
                self.model.add(Flatten())
                self.model.add(Dense(cl['dense'], activation='hard_sigmoid'))
                total_size = cl['dense']
                break

        assert total_size % self.sentence_len == 0, \
            'Total size %d not divisible to sentence length %d' % (total_size, self.sentence_len)
        rnn_depth = int(total_size / self.sentence_len)

        self.model.add(Reshape((self.sentence_len, rnn_depth)))
        for i, ll in enumerate(self.lstm_layers):
            self.model.add(LSTM(ll['units'], activation='tanh', return_sequences=True,
                                recurrent_activation='hard_sigmoid', dropout=self.dropout,
                                recurrent_dropout=self.dropout, implementation=2, unroll=False))

        self.model.add(LSTM(self.vocab.size, activation='softmax', return_sequences=True,
                            recurrent_activation='hard_sigmoid', dropout=self.dropout,
                            recurrent_dropout=self.dropout, implementation=2, unroll=False))
