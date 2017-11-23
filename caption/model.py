from keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from recurrentshop import RecurrentSequential, LSTMCell

import math
import numpy as np

from . import utils


class CaptionModel:
    def __init__(self, conv_layers, lstm_layers, img_size=(128, 128),
                 sentence_len=20, save_dir='models/default'):
        self.img_size = img_size
        self.conv_layers = conv_layers
        self.lstm_layers = lstm_layers
        self.sentence_len = sentence_len
        self.save_dir = save_dir

        self.seq_model = None
        self.model = Sequential()

    def build(self, vocab_size, readout=False):
        self.seq_model = Sequential()

        out_row, out_col = self.img_size
        conv_depth = None
        for i, cl in enumerate(self.conv_layers):
            if i == 0:
                self.seq_model.add(Conv2D(
                    cl['filters'], cl['kernel'], strides=cl['strides'],
                    activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3),
                    padding='same'
                ))
            else:
                self.seq_model.add(Conv2D(
                    cl['filters'], cl['kernel'], strides=cl['strides'],
                    activation='relu', padding='same'
                ))
            if 'pool' in cl.keys():
                self.seq_model.add(MaxPooling2D(cl['pool']))
                out_col /= cl['pool']
                out_row /= cl['pool']
            if 'dense' in cl.keys():
                self.seq_model.add(Dense(cl['dense'], activation='hard_sigmoid'))
                conv_depth = cl['dense']
                break

        conv_depth = self.conv_layers[-1]['filters'] if conv_depth is None else conv_depth
        self.seq_model.add(Reshape((int(out_row * out_col), conv_depth)))

        for i, ll in enumerate(self.lstm_layers[:-1]):
            self.seq_model.add(LSTM(ll['units'], activation='tanh', return_sequences=True,
                                    recurrent_activation='hard_sigmoid'))
        self.seq_model.add(LSTM(self.lstm_layers[-1]['units'], activation='tanh', return_sequences=False,
                                recurrent_activation='hard_sigmoid'))

        if readout:
            assert self.lstm_layers[-1]['units'] == vocab_size
        rnn = RecurrentSequential(decode=True, output_length=self.sentence_len,
                                  readout=readout)
        rnn.add(LSTMCell(vocab_size, activation='softmax'))
        self.seq_model.add(rnn)
        self.model = self.seq_model

    def compile(self, optimizer=RMSprop()):
        self.model.compile(optimizer, loss='categorical_crossentropy',
                           metrics=['acc'], sample_weight_mode='temporal')

    def train(self, img_paths, captions, val_img, val_captions, vocab,
              epochs=100, batch_size=10, initial_epoch=0):
        per_epoch = math.ceil(len(captions) / batch_size)
        val_per_epoch = math.ceil(len(val_captions) / batch_size)

        self.model.fit_generator(self._generate_batch(img_paths, captions, vocab, batch_size),
                                 per_epoch, shuffle=False, max_queue_size=3, epochs=epochs, initial_epoch=initial_epoch,
                                 validation_data=self._generate_batch(val_img, val_captions, vocab, batch_size),
                                 validation_steps=val_per_epoch)

    def _generate_batch(self, img_paths, captions, vocab, batch_size):
        while True:
            for i in range(batch_size, len(captions), batch_size):
                _img_paths = img_paths[i-batch_size:i]
                _captions = captions[i-batch_size:i]

                _img_mat = np.zeros((batch_size, self.img_size, self.img_size, 3))
                for j, _path in enumerate(_img_paths):
                    _img_mat[j, :, :] = utils.load_image(_path, size=self.img_size)

                _cap_mat = np.zeros((batch_size, self.sentence_len, vocab.size))
                _cap_mask = np.zeros((batch_size, self.sentence_len, 1))
                for j, _caption in enumerate(_captions):
                    _cap_mat[j, :, :] = vocab.encode_sentence(_caption, length=self.sentence_len)
                    _cap_mask[j, :, :] = vocab.mask_sentence(_caption, length=self.sentence_len)

                yield _img_mat, _cap_mat, _cap_mask

    def summary(self):
        self.model.summary()

